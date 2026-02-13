import numpy as np
import torch
# from pytorch_msssim import ssim
from monai.metrics import SSIMMetric

# Note that we use the axial slice-wise performance metrics to evaluate all competing methods
# and ours because the axial axis is mainly used in real clinical scenarios.

# def mean_absolute_error(image_true, image_generated):
#     """Compute mean absolute error.

#     Args:
#         image_true: (Tensor) true image
#         image_generated: (Tensor) generated image

#     Returns:
#         mse: (float) mean squared error
#     """
#     image_true = image_true.squeeze(0).squeeze(0)
#     image_generated = image_generated.squeeze(0).squeeze(0)
#     losses = 0.
#     depth = image_true.shape[0]
#     for i in range(depth):
#         losses += torch.abs(image_true[i, :, :] - image_generated[i, :, :]).mean()
#     losses /= depth
#     return losses


# def peak_signal_to_noise_ratio(image_true, image_generated):
#     """"Compute peak signal-to-noise ratio.

#     Args:
#         image_true: (Tensor) true image
#         image_generated: (Tensor) generated image

#     Returns:
#         psnr: (float) peak signal-to-noise ratio"""
#     image_true = image_true.squeeze(0).squeeze(0)
#     image_generated = image_generated.squeeze(0).squeeze(0)
#     losses = 0.0
#     depth = image_true.shape[0]
#     for i in range(depth):
#         mse = ((image_true[i, :, :] - image_generated[i, :, :]) ** 2).mean()
#         # Numerical stability: when mse == 0, PSNR is +inf; clamp avoids log10(0) warnings.
#         eps = torch.finfo(mse.dtype).eps
#         mse = torch.clamp(mse, min=eps)
#         data_range = 1.0
#         psnr = 10.0 * torch.log10((data_range ** 2) / mse)
#         losses += psnr
#     losses = losses / float(depth)
#     return losses


# # def structural_similarity_index(image_true, image_generated):
# #     """Compute structural similarity index.

# #     Args:
# #         image_true: (Tensor) true image
# #         image_generated: (Tensor) generated image

# #     Returns:
# #         ssim: (float) structural similarity index """
# #     image_true = image_true.squeeze(0).squeeze(0)
# #     image_generated = image_generated.squeeze(0).squeeze(0)
# #     losses = 0.
# #     depth = image_true.shape[0]
# #     for i in range(depth):
# #         losses += ssim(image_generated[i, :, :].unsqueeze(0).unsqueeze(0),
# #                        image_true[i, :, :].unsqueeze(0).unsqueeze(0), size_average=True)
# #     losses /= depth
# #     return losses

# import numpy as np
# import torch
# from monai.losses import SSIMLoss

# def structural_similarity_index(image_true, image_generated, data_range=1.0, win_size=11):
#     """
#     Axial slice-wise SSIM using MONAI.
#     Returns average SSIM over depth.

#     Assumes input shape: [1, 1, D, H, W] or compatible.
#     """
#     # [D, H, W]
#     image_true = image_true.squeeze(0).squeeze(0)
#     image_generated = image_generated.squeeze(0).squeeze(0)

#     # stack axial slices as a batch: [D, 1, H, W]
#     x = image_generated.unsqueeze(1)
#     y = image_true.unsqueeze(1)

#     # MONAI SSIMLoss: returns (1 - SSIM)
#     ssim_loss = SSIMLoss(
#         spatial_dims=2,
#         data_range=data_range,
#         win_size=win_size,
#         reduction="mean",
#     ).to(x.device)

#     loss = ssim_loss(x, y)          # scalar
#     ssim_value = 1.0 - loss         # SSIM metric

#     return ssim_value
import torch
import torch.nn.functional as F
from monai.metrics import SSIMMetric

def mean_absolute_error(image_true, image_generated):
    """
    计算 MAE (Mean Absolute Error)。
    支持任意维度输入 (B, C, D, H, W)，计算所有元素的平均绝对误差。
    """
    return torch.abs(image_true - image_generated).mean()

def peak_signal_to_noise_ratio(image_true, image_generated, data_range=1.0):
    """
    计算 3D PSNR。
    逻辑：先计算每个样本的 MSE，再转为 PSNR，最后对 Batch 取平均。
    输入形状应为 (B, C, D, H, W)。
    """
    # 确保输入是 5 维 (B, C, D, H, W)
    if image_true.dim() == 4: # 如果是 (C, D, H, W) -> 增加 Batch 维
        image_true = image_true.unsqueeze(0)
        image_generated = image_generated.unsqueeze(0)
    
    # 1. 计算每个样本的 MSE (在 D, H, W, C 维度上取平均，保留 Batch 维度)
    # dim=(1, 2, 3, 4) 表示在 (C, D, H, W) 上求平均
    mse = torch.mean((image_true - image_generated) ** 2, dim=(1, 2, 3, 4))
    
    # 2. 数值稳定性处理 (防止 MSE 为 0 导致 log(0))
    eps = torch.finfo(mse.dtype).eps
    mse = torch.clamp(mse, min=eps)
    
    # 3. 计算每个样本的 PSNR
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    
    # 4. 返回 Batch 的平均 PSNR
    return psnr.mean()

def structural_similarity_index(image_true, image_generated, data_range=1.0):
    """
    计算 3D SSIM。
    使用 MONAI 的 SSIMMetric (不是 Loss)，直接计算 3D 结构相似度。
    输入形状应为 (B, C, D, H, W)。
    """
    # 确保输入是 5 维
    if image_true.dim() == 4:
        image_true = image_true.unsqueeze(0)
        image_generated = image_generated.unsqueeze(0)

    # 初始化 MONAI 的 SSIM 计算器
    # spatial_dims=3 表示计算 3D SSIM (不再是 2D 切片平均)
    ssim_metric = SSIMMetric(spatial_dims=3, data_range=data_range, win_size=7)
    
    # MONAI 需要将数据移动到同一设备
    device = image_true.device
    
    # 计算 SSIM
    # y_pred=generated, y=true
    ssim_val = ssim_metric(y_pred=image_generated, y=image_true)
    
    # ssim_metric 返回的是 (B, 1) 的 tensor，我们需要平均值
    return ssim_val.mean()

# import torch
# import torch.nn.functional as F
# from monai.metrics import SSIMMetric

# def masked_mean_absolute_error(image_true, image_generated, mask):
#     """
#     支持 Batch 模式的 Masked MAE
#     输入形状: (B, C, D, H, W)
#     """
#     mask = (mask > 0).float()
#     # 1. 计算每个像素的绝对误差
#     abs_diff = torch.abs(image_true - image_generated) * mask
    
#     # 2. 对每个样本独立求和 (dim=1,2,3,4 代表 C,D,H,W)
#     sum_error_per_sample = abs_diff.sum(dim=(1, 2, 3, 4))
#     sum_mask_per_sample = mask.sum(dim=(1, 2, 3, 4))
    
#     # 3. 计算每个样本的 MAE，然后取 Batch 平均
#     mae_per_sample = sum_error_per_sample / (sum_mask_per_sample + 1e-8)
#     return mae_per_sample.mean()

# def masked_peak_signal_to_noise_ratio(image_true, image_generated, mask, data_range=1.0):
#     """
#     支持 Batch 模式的 Masked PSNR
#     """
#     mask = (mask > 0).float()
#     # 1. 计算每个样本在 Mask 内的 MSE
#     mse_map = torch.pow(image_true - image_generated, 2) * mask
#     mse_per_sample = mse_map.sum(dim=(1, 2, 3, 4)) / (mask.sum(dim=(1, 2, 3, 4)) + 1e-8)
    
#     # 2. 数值稳定性
#     eps = torch.finfo(mse_per_sample.dtype).eps
#     mse_per_sample = torch.clamp(mse_per_sample, min=eps)
    
#     # 3. 计算每个样本的 PSNR 并平均
#     psnr_per_sample = 10.0 * torch.log10((data_range ** 2) / mse_per_sample)
#     return psnr_per_sample.mean()

# def masked_structural_similarity_index(image_true, image_generated, mask, data_range=1.0):
#     """
#     支持 Batch 模式的 Masked SSIM
#     """
#     mask = (mask > 0).float()
#     # 抹除背景干扰
#     image_true_masked = image_true * mask
#     image_generated_masked = image_generated * mask

#     # MONAI 的 SSIMMetric 天生支持 Batch，返回 (B, 1) 的张量
#     ssim_metric = SSIMMetric(spatial_dims=3, data_range=data_range, win_size=7)
    
#     # 确保维度对齐
#     if image_true_masked.dim() == 4:
#         image_true_masked = image_true_masked.unsqueeze(0)
#         image_generated_masked = image_generated_masked.unsqueeze(0)

#     ssim_values = ssim_metric(y_pred=image_generated_masked, y=image_true_masked)
#     return ssim_values.mean() # 返回 Batch 的平均值


from monai.losses import SSIMLoss

def structural_similarity_index_mask(image_true, image_generated, data_range=1.0, mask=None):
    """
    计算 3D SSIM。
    如果提供了 mask，则只计算 mask=1 区域内的平均 SSIM。
    """
    # 维度调整
    if image_true.dim() == 4:
        image_true = image_true.unsqueeze(0)
        image_generated = image_generated.unsqueeze(0)
    if mask is not None and mask.dim() == 4:
        mask = mask.unsqueeze(0)

    # 1. 使用 SSIMLoss 获取 SSIM Map (不进行平均 reduction="none")
    # MONAI 的 SSIMLoss 计算的是 1 - SSIM，所以我们要反过来
    ssim_loss_func = SSIMLoss(spatial_dims=3, data_range=data_range, win_size=7, reduction="none")
    
    # loss_map 的每个像素值是 1 - ssim_value
    loss_map = ssim_loss_func(image_generated, image_true)
    
    # 得到 ssim_map (每个体素的相似度)
    ssim_map = 1.0 - loss_map

    # 2. 根据 Mask 计算平均值
    if mask is not None:
        # SSIM 是卷积计算，边缘会缩减，所以 mask 也要适配 SSIM map 的尺寸
        # 如果 ssim_map 尺寸和 mask 不一样（通常不会），需要注意。
        # MONAI 的 SSIM 默认 padding='reflection'，尺寸应该保持一致。
        
        # 只取 mask 区域的平均
        masked_ssim = ssim_map * mask
        return masked_ssim.sum() / (mask.sum() + 1e-8)
    else:
        # 全图平均
        return ssim_map.mean()