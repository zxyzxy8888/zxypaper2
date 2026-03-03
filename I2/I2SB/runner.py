# ---------------------------------------------------------------
# Runner for 3D MRI-to-PET Generation
# Adapted for Custom Dataset & 3D Volumetric Data
# ---------------------------------------------------------------

import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
from itertools import cycle
from pathlib import Path

# 引入项目模块
try:
    from . import util
    # 确保你的 network.py 中类名是 Image3DNet (根据上一轮修改)
    from .network import Image3DNet
    from .diffusion import Diffusion
    from .performance_metric import (
        mean_absolute_error,
        peak_signal_to_noise_ratio,
        structural_similarity_index,
    )
except Exception:
    import util
    from network import Image3DNet
    from diffusion import Diffusion
    from performance_metric import (
        mean_absolute_error,
        peak_signal_to_noise_ratio,
        structural_similarity_index,
    )

try:
    import distributed_util as dist_util
except Exception:
    dist_util = None

# -----------------------------------------------------------------------------
# 辅助函数 (Data Pre/Post Processing)
# -----------------------------------------------------------------------------

def get_mid_slice(x):
    """
    取 3D 数据的中间切片用于 Tensorboard 可视化
    Input: [B, C, D, H, W] -> Output: [B, C, H, W]
    """
    if x is None: return None
    # 取深度(Depth)维度的中间张
    mid = x.shape[2] // 2
    return x[:, :, mid, :, :]

def process_data(mri, pet, target_size=(96, 96, 96)):
    """
    数据预处理：
    1. 插值 (Resize) 到目标尺寸 (例如 96x96x96)
    2. 归一化：从 [0, 1] 映射到 [-1, 1] 以适配扩散模型
    """
    # 1. 插值 (Trilinear for 3D)
    # 输入维度应为 [B, C, D, H, W]
    if mri.shape[2:] != target_size:
        # align_corners=False 是默认且推荐的设置
        mri = F.interpolate(mri, size=target_size, mode='trilinear', align_corners=False)
        pet = F.interpolate(pet, size=target_size, mode='trilinear', align_corners=False)
    
    # 2. 归一化 [0, 1] -> [-1, 1]
    # 假设 Dataset 输出的是 0-1
    mri = mri * 2.0 - 1.0
    pet = pet * 2.0 - 1.0
    
    return mri, pet

def denormalize(x):
    """ 
    反归一化：[-1, 1] -> [0, 1]
    并进行截断(clamp)防止数值溢出，用于计算指标
    """
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)

def build_optimizer_sched(opt, net, log):
    """ 构建优化器和学习率调度器 """
    optim_dict = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    if opt.lr_gamma < 1.0:
        sched_dict = {"step_size": opt.lr_step, 'gamma': opt.lr_gamma}
        sched = lr_scheduler.StepLR(optimizer, **sched_dict)
    else:
        sched = None

    # 断点续训加载
    if opt.load and os.path.exists(opt.load):
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer from {opt.load}")
        if sched is not None and "sched" in checkpoint:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded scheduler from {opt.load}")

    return optimizer, sched

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    """ 生成扩散模型的 Beta 调度表 (I2SB 风格) """
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return betas.numpy()

# -----------------------------------------------------------------------------
# Runner Class
# -----------------------------------------------------------------------------

class Runner(object):
    def __init__(self, opt, log):
        super(Runner, self).__init__()
        self.opt = opt
        self.log = log
        self.best_mae = float('inf')  # 记录验证集最优 MAE
        # 强制目标 3D 尺寸
        self.target_size = (96, 96, 96) 

        # 1. 保存配置参数
        if opt.global_rank == 0:
            os.makedirs(opt.ckpt_path, exist_ok=True)
            opt_pkl_path = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        # 2. 构建 Diffusion (Beta Schedule)
        # 对称 Beta 调度，长度严格等于 opt.interval（奇偶都兼容）
        betas = make_beta_schedule(n_timestep=opt.interval, linear_end=opt.beta_max / opt.interval)
        half = (opt.interval + 1) // 2
        betas = np.concatenate([betas[:half], np.flip(betas[:half])])[:opt.interval]
        self.diffusion = Diffusion(betas, opt.device)
        log.info(f"[Diffusion] Initialized steps={len(betas)}")

        # 3. 构建 3D 网络
        noise_levels = torch.linspace(opt.t0, opt.T, opt.interval, device=opt.device) * opt.interval
        
        self.net = Image3DNet(
            log, 
            noise_levels=noise_levels, 
            use_fp16=opt.use_fp16, 
            cond=False,         # 标准 I2SB：不使用额外 cond 分支（MRI 作为端点 x1）
            image_size=96       # 对应 target_size
        )
        
        # 4. 初始化 EMA (指数移动平均，稳定生成质量)
        self.ema = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)

        # 5. 加载权重 (Resume)
        if opt.load and os.path.exists(opt.load):
            checkpoint = torch.load(opt.load, map_location="cpu")
            state_dict = checkpoint['net']
            # 处理 DDP 可能带来的 'module.' 前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.net.load_state_dict(new_state_dict)
            log.info(f"[Net] Loaded weights from {opt.load}")
            
            if "ema" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema"])

        self.net.to(opt.device)
        self.ema.to(opt.device)

    def _unpack_batch(self, batch):
        """兼容不同 Dataset 返回格式，统一提取 (mri, pet)。"""
        if isinstance(batch, (list, tuple)):
            if len(batch) < 2:
                raise ValueError("Batch must contain at least MRI and PET tensors.")
            return batch[0], batch[1]
        raise TypeError("Unsupported batch format. Expected tuple/list with MRI and PET.")

    def _eval_nfe(self, opt, default_nfe=49):
        """验证/测试的采样步数控制：优先使用配置，否则采用较快默认值。"""
        if hasattr(opt, "eval_nfe") and opt.eval_nfe is not None:
            return int(min(max(opt.eval_nfe, 1), opt.interval - 1))
        return int(min(default_nfe, opt.interval - 1))

    def _compute_metrics(self, pred_01, gt_01):
        """在 [0,1] 空间计算 MAE / PSNR / SSIM。"""
        mae = mean_absolute_error(gt_01, pred_01)
        psnr = peak_signal_to_noise_ratio(gt_01, pred_01)
        ssim = structural_similarity_index(gt_01, pred_01)
        return mae, psnr, ssim

    def compute_label(self, step, x0, xt):
        """ 计算训练 Loss 的目标 (Eq 12) """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ 采样时使用：从网络输出恢复 x0 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def save_checkpoint(self, filename, net, optimizer, sched):
        """ 保存模型权重 """
        save_dict = {
            "net": net.module.state_dict() if isinstance(net, DDP) else net.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
        }
        torch.save(save_dict, self.opt.ckpt_path / filename)

    # -------------------------------------------------------------------------
    # Train Function
    # -------------------------------------------------------------------------
    def train(self, opt, train_loader, val_loader):
        self.writer = util.build_log_writer(opt)  # TensorBoard Writer
        log = self.log

        if opt.distributed:
            net = DDP(self.net, device_ids=[opt.device])
        else:
            net = self.net
            
        optimizer, sched = build_optimizer_sched(opt, net, log)
        
        # 将 DataLoader 转为无限循环迭代器
        train_iter = iter(train_loader)

        log.info(f"Start training for {opt.num_itr} iterations...")
        net.train()
        
        for it in range(opt.num_itr):
            optimizer.zero_grad()

            # --- 1. 获取数据 ---
            # 直接从 DataLoader 获取 (MRI, PET)
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            mri, pet = self._unpack_batch(batch)

            mri = mri.to(opt.device)
            pet = pet.to(opt.device)

            # --- 2. 预处理 (插值到96 + 归一化-1~1) ---
            mri, pet = process_data(mri, pet, self.target_size)
            
            x0 = pet   # 目标端点 x0 (PET) [-1, 1]
            x1 = mri   # 源端点  x1 (MRI) [-1, 1]，标准 I2SB

            # --- 3. 扩散前向过程 ---
            step = torch.randint(0, opt.interval, (x0.shape[0],), device=opt.device)
            
            # q_sample: 将端点 x0(PET) 和 x1(MRI) 混合得到 xt
            xt = self.diffusion.q_sample(step, x0, x1, ot_ode=opt.ot_ode)
            
            # 计算回归目标
            label = self.compute_label(step, x0, xt)

            # --- 4. 网络预测 ---
            # 标准 I2SB：网络输入仅使用 xt 与 step（不额外输入 cond）
            pred = net(xt, step)
            
            # --- 5. 反向传播 ---
            loss = F.mse_loss(pred, label)
            loss.backward()
            optimizer.step()
            self.ema.update()
            if sched is not None: sched.step()

            # --- 6. 日志记录 ---
            if it % 100 == 0:
                log.info(f"Iter {it}/{opt.num_itr} | LR: {optimizer.param_groups[0]['lr']:.2e} | Loss: {loss.item():.5f}")
                self.writer.add_scalar(it, 'train/loss', loss.detach())

            # 定期保存最新模型 (latest)
            if it % 5000 == 0 and it > 0:
                if opt.global_rank == 0:
                    self.save_checkpoint("latest.pt", net, optimizer, sched)
                    log.info(f"Saved latest checkpoint at iter {it}")

            # 定期验证 (Validation)
            # 建议频率设置低一些，因为 3D 采样很慢
            if it % 2000 == 0 and it > 0:
            # if it % 2000 == 0:
                val_stats = self.validate(opt, it, val_loader)
                val_mae = val_stats["mae"]
                
                # 仅在主进程保存最优模型
                if opt.global_rank == 0:
                    if val_mae < self.best_mae:
                        self.best_mae = val_mae
                        self.save_checkpoint("best_mae.pt", net, optimizer, sched)
                        log.info(f"★ New Best MAE: {self.best_mae:.4f} saved!")
                
                net.train() # 恢复训练模式

        self.writer.close()

    # -------------------------------------------------------------------------
    # Validation Function
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, opt, it, val_loader):
        self.log.info(f"=== Validating at iter {it} ===")
        self.net.eval()
        
        total_mae = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        eval_nfe = self._eval_nfe(opt, default_nfe=49)
        
        # 使用 EMA 权重进行生成，质量更好
        with self.ema.average_parameters():
            for i, batch in enumerate(val_loader):
                # 验证前 5 个 Batch 即可，节省时间
                # if i >= 5: break 

                mri, pet = self._unpack_batch(batch)
                
                mri = mri.to(opt.device)
                pet = pet.to(opt.device)

                # 预处理
                mri, pet = process_data(mri, pet, self.target_size)
                
                # 采样起点：标准 I2SB 使用源端点 MRI 作为 x1
                x1 = mri
                
                # 执行采样
                # ddpm_sampling 返回轨迹列表，我们取最后一个 [-1] 即最终结果
                _, pred_pet_traj = self.ddpm_sampling(opt, x1, nfe=eval_nfe, verbose=False)
                pred_pet = pred_pet_traj[:, -1].to(opt.device) # Shape: [B, 1, D, H, W]
                
                # 反归一化到 [0, 1] 用于计算指标
                pred_01 = denormalize(pred_pet)
                pet_01 = denormalize(pet)

                mae, psnr, ssim = self._compute_metrics(pred_01, pet_01)
                total_mae += mae.item()
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                count += 1
                
                # TensorBoard 可视化 (只记录第一张图)
                if i == 0 and opt.global_rank == 0:
                    self.log_visuals(it, mri, pet, pred_pet, suffix="val")

        avg_mae = total_mae / max(count, 1)
        avg_psnr = total_psnr / max(count, 1)
        avg_ssim = total_ssim / max(count, 1)
        
        if opt.global_rank == 0:
            self.writer.add_scalar(it, 'val/MAE', avg_mae)
            self.writer.add_scalar(it, 'val/PSNR', avg_psnr)
            self.writer.add_scalar(it, 'val/SSIM', avg_ssim)
            self.log.info(
                f"Validation (nfe={eval_nfe}) | MAE: {avg_mae:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}"
            )
            
        return {"mae": avg_mae, "psnr": avg_psnr, "ssim": avg_ssim}

    def log_visuals(self, it, mri, pet, pred, suffix=""):
        """ 记录切片图像到 TensorBoard """
        # 反归一化用于显示
        mri = denormalize(mri)
        pet = denormalize(pet)
        pred = denormalize(pred)

        # 取中间切片: [B, 1, D, H, W] -> [B, 1, H, W]
        sl_mri = get_mid_slice(mri)
        sl_pet = get_mid_slice(pet)
        sl_pred = get_mid_slice(pred)
        
        # 拼接: MRI | Real | Fake
        # 只取 Batch 中的第一个样本
        display = torch.cat([sl_mri[0:1], sl_pet[0:1], sl_pred[0:1]], dim=0)
        
        # 制作网格
        grid = tu.make_grid(display, nrow=3)
        self.writer.add_image(it, f'visuals/{suffix}', grid)

    # -------------------------------------------------------------------------
    # Test Function
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def test(self, test_loader):
        opt = self.opt
        best_ckpt = opt.ckpt_path / "best_mae.pt"
        
        # 加载最优权重
        if not os.path.exists(best_ckpt):
            self.log.warning("No best checkpoint found! Using current weights.")
        else:
            checkpoint = torch.load(best_ckpt, map_location=opt.device)
            if "ema" in checkpoint:
                self.ema.load_state_dict(checkpoint["ema"])
                self.log.info("Loaded BEST EMA weights for testing.")
            else:
                new_sd = {k.replace('module.', ''): v for k, v in checkpoint['net'].items()}
                self.net.load_state_dict(new_sd)
                self.log.info("Loaded BEST Net weights for testing.")

        self.net.eval()
        self.log.info("=== Starting Test on Test Set ===")
        
        total_mae = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        n_batches = 0
        eval_nfe = self._eval_nfe(opt, default_nfe=opt.interval - 1)
        
        with self.ema.average_parameters():
            for i, batch in enumerate(test_loader):
                mri, pet = self._unpack_batch(batch)
                mri = mri.to(opt.device)
                pet = pet.to(opt.device)
                
                # 预处理
                mri, pet = process_data(mri, pet, self.target_size)
                
                # 采样起点
                x1 = mri
                
                # 采样
                _, pred_traj = self.ddpm_sampling(opt, x1, nfe=eval_nfe, verbose=False)
                pred_pet = pred_traj[:, -1].to(opt.device)
                
                # 指标计算 (0-1 空间)
                pred_01 = denormalize(pred_pet)
                pet_01 = denormalize(pet)

                mae, psnr, ssim = self._compute_metrics(pred_01, pet_01)
                
                total_mae += mae.item()
                total_psnr += psnr.item()
                total_ssim += ssim.item()
                n_batches += 1
                
                if i % 10 == 0:
                    self.log.info(
                        f"Test Batch {i}: MAE={mae.item():.4f}, PSNR={psnr.item():.2f}, SSIM={ssim.item():.4f}"
                    )

        avg_mae = total_mae / max(n_batches, 1)
        avg_psnr = total_psnr / max(n_batches, 1)
        avg_ssim = total_ssim / max(n_batches, 1)
        
        self.log.info(f"=== Final Test Results ===")
        self.log.info(f"nfe:      {eval_nfe}")
        self.log.info(f"Avg MAE:  {avg_mae:.4f}")
        self.log.info(f"Avg PSNR: {avg_psnr:.4f}")
        self.log.info(f"Avg SSIM: {avg_ssim:.4f}")

    # -------------------------------------------------------------------------
    # DDPM Sampling Core
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def ddpm_sampling(self, opt, x1, mask=None, clip_denoise=True, nfe=None, log_count=10, verbose=True):
        """ 执行推理生成 """
        nfe = nfe or opt.interval-1
        steps = util.space_indices(opt.interval, nfe+1)
        log_count = min(len(steps)-1, log_count)
        log_steps = [steps[i] for i in util.space_indices(len(steps)-1, log_count)]
        
        if verbose:
            self.log.info(f"[Sampling] steps={opt.interval}, nfe={nfe}")

        x1 = x1.to(opt.device)

        with self.ema.average_parameters():
            self.net.eval()

            def pred_x0_fn(xt, step):
                # 构造时间步 Batch
                step_ts = torch.full((xt.shape[0],), step, device=opt.device, dtype=torch.long)
                # 网络预测（标准 I2SB: 无额外 cond）
                out = self.net(xt, step_ts)
                return self.compute_pred_x0(step, xt, out, clip_denoise=clip_denoise)

            xs, pred_x0 = self.diffusion.ddpm_sampling(
                steps, pred_x0_fn, x1, mask=mask, ot_ode=opt.ot_ode, log_steps=log_steps, verbose=verbose,
            )

        return xs, pred_x0