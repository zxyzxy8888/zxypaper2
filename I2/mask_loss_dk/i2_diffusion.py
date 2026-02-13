import torch
import numpy as np
from tqdm import tqdm
from functools import partial

# -------------------------------------------------------------------------
# Part 1: 核心数学工具函数
# -------------------------------------------------------------------------

def unsqueeze_xdim(z, xdim):
    """
    让时间步参数 z (Batch,) 适配 3D 张量 (Batch, C, D, H, W)
    """
    # xdim 是 (Batch, C, D, H, W)，长度为 5
    # 我们需要将 z 扩展为 (Batch, 1, 1, 1, 1)
    dims_to_add = len(xdim) - 1
    shape = z.shape + (1,) * dims_to_add

    return z.reshape(shape)

def compute_gaussian_product_coef(sigma1, sigma2):
    """
    I2SB 核心公式：计算两个高斯分布乘积的系数。
    Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
    return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) 
    """
    denom = sigma1**2 + sigma2**2
    # 防止除以0 (虽然在扩散过程中一般不会发生，但加上eps更稳健，这里保持原版逻辑)
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var

# -------------------------------------------------------------------------
# Part 2: Diffusion 类 (I2SB 调度器)
# -------------------------------------------------------------------------

class Diffusion():
    def __init__(self, betas, device):
        self.device = device

        # 1. 计算前向累积噪声标准差 (Forward Std) -> 从 x0 到 xt
        std_fwd = np.sqrt(np.cumsum(betas))
        
        # 2. 计算后向累积噪声标准差 (Backward Std) -> 从 x1 到 xt
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))
        
        # 3. 计算薛定谔桥 (Schrödinger Bridge) 的混合系数
        # mu_x0: PET 的权重
        # mu_x1: MRI 的权重
        # var: 桥接过程的方差
        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)
        std_sb = np.sqrt(var)

        # 4. 转为 Tensor 并存入 GPU
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to(device)
        self.std_fwd = to_torch(std_fwd).to(device)
        self.std_bwd = to_torch(std_bwd).to(device)
        self.std_sb  = to_torch(std_sb).to(device)
        self.mu_x0 = to_torch(mu_x0).to(device)
        self.mu_x1 = to_torch(mu_x1).to(device)

    def q_sample(self, step, x0, x1, ot_ode=False):
        """
        [训练阶段核心] 前向加噪过程
        Sample q(x_t | x_0, x_1)
        
        x0: 目标图像 (PET), shape (B, C, D, H, W)
        x1: 条件/源图像 (MRI), shape (B, C, D, H, W)
        step: 当前时间步 t
        """
        assert x0.shape == x1.shape
        batch_size = x0.shape[0]
        # 获取除了 batch 以外的维度信息用于 unsqueeze
        xdim = x0.shape 

        # 获取当前步的系数，并扩展维度以适配 3D
        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        # 核心混合公式: xt 是 x0 和 x1 的加权平均
        xt = mu_x0 * x0 + mu_x1 * x1
        
        # 加上随机噪声 (如果是 ODE 模式则不加，但一般训练都加)
        if not ot_ode:
            xt = xt + std_sb * torch.randn_like(xt)
            
        return xt.detach()

    def p_posterior(self, nprev, n, x_n, x0_pred, ot_ode=False):
        """
        [推理阶段单步] 后验采样，计算 x_{t-1}
        Sample p(x_{nprev} | x_n, x_0_pred)
        
        nprev: 下一个时间步 (t-1)
        n: 当前时间步 (t)
        x_n: 当前的噪声图 x_t
        x0_pred: 模型预测出来的干净 PET
        """
        # 获取数据维度
        xdim = x_n.shape

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        
        # 计算这一步的标准差变化量
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        # 再次利用高斯乘积公式计算后验均值系数
        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        # 扩展维度
        mu_x0 = unsqueeze_xdim(mu_x0, xdim)
        mu_xn = unsqueeze_xdim(mu_xn, xdim)
        var   = unsqueeze_xdim(var,   xdim)

        # 计算下一步的均值: xt_prev_mean
        xt_prev = mu_x0 * x0_pred + mu_xn * x_n
        
        # 注入 Langevin 噪声 (除非是 ODE 模式或最后一步)
        if not ot_ode and nprev > 0:
            xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    def ddpm_sampling(self, steps, pred_x0_fn, x1, mask=None, ot_ode=False, verbose=True):
        """
        [推理阶段循环] 完整的生成过程
        
        steps: 时间步列表，例如 [999, 998, ... 0]
        pred_x0_fn: 一个函数，输入 (xt, t)，输出预测的 x0
        x1: 采样的起点 (Source Image, 即 MRI)
        """
        # 1. 初始化: xt 从 x1 (MRI) 开始
        xt = x1.detach().to(self.device)

        xs = []       # 记录轨迹 (可选)
        pred_x0s = [] # 记录预测的 x0 (可选)

        # 确保步数是倒序的 (从 T 到 0)
        # steps[0] 应该是 0 (终点), steps[-1] 应该是 T (起点)
        # 这里逻辑稍微有点绕，I2SB 原版是把 steps 列表传进来，通常是 list(range(1000))
        # 我们需要把它倒过来遍历
        
        # 构造 (prev_step, step) 对，例如 (998, 999), (997, 998)...
        # 注意：steps 输入进来是 [0, 1, 2... T-1]
        
        steps = steps[::-1] # 翻转: [T-1, ..., 2, 1, 0]

        pair_steps = zip(steps[1:], steps[:-1]) # (T-2, T-1), ..., (0, 1)
        
        if verbose:
            pair_steps = tqdm(pair_steps, desc='I2SB Sampling', total=len(steps)-1)
            
        for prev_step, step in pair_steps:
            # step 是当前步 (大), prev_step 是下一步 (小)
            
            # 2. 模型预测: 看着当前的 xt, 猜 x0 (PET)
            # 注意: pred_x0_fn 内部会处理 condition (MRI) 的拼接
            pred_x0 = pred_x0_fn(xt, torch.tensor([step], device=self.device))
            
            # 3. 更新 xt: 往 x0 的方向走一步
            xt = self.p_posterior(prev_step, step, xt, pred_x0, ot_ode=ot_ode)

            # (可选) 记录中间结果
            # pred_x0s.append(pred_x0.detach().cpu())
            # xs.append(xt.detach().cpu())

        # 这里的返回值结构你可以根据需要改
        # 为了兼容你之前的 main.py，我返回一个 dummy list 包裹最终结果
        # 你的 main.py 里取的是 traj_x0[:, 0]，所以我们把最终结果放进去
        
        # 重新预测一次 x0 作为最终输出 (此时 xt 已经是 t=0 时刻的了)
        final_pred_x0 = pred_x0_fn(xt, torch.tensor([0], device=self.device))
        
        # 为了匹配你 main.py 的接口: return stack_bwd_traj(xs), stack_bwd_traj(pred_x0s)
        # 我们构造一个假的 list，只包含最终结果
        # 维度: (B, 1, C, D, H, W)
        return xt.unsqueeze(1), final_pred_x0.unsqueeze(1)