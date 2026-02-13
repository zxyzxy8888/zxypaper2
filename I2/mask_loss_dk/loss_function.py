# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch.autograd import Variable
import numpy as np
from math import exp
# from pytorch_msssim import SSIM
from monai.losses import SSIMLoss, PerceptualLoss
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(pred, target)


# ----------------------------
# 2. SSIM Loss
# ----------------------------

def gaussian(window_size, sigma):
    x = torch.arange(window_size, dtype=torch.float32)
    gauss = torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    pad = window_size // 2
    img1_padded = F.pad(img1, (pad, pad, pad, pad), mode='reflect')
    img2_padded = F.pad(img2, (pad, pad, pad, pad), mode='reflect')
    mu1 = F.conv2d(img1_padded, window, padding = 0, groups = channel)
    mu2 = F.conv2d(img2_padded, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1_padded*img1_padded, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2_padded*img2_padded, window, padding = 0, groups = channel) - mu2_sq
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    sigma12 = F.conv2d(img1_padded*img2_padded, window, padding = 0, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1,2,3])
    
def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    # print(f"img1 data range: {img1.min()} to {img1.max()}, img2 data range: {img2.min()} to {img2.max()}")
    # # print(f"img1 {img1}, img2 {img2}")
    # print(f"img1 shape: {img1.shape}, img2 shape: {img2.shape}")
    # print(f"img1 non-zero count: {torch.count_nonzero(img1)} / {img1.numel()}")
    # print(f"img2 non-zero count: {torch.count_nonzero(img2)} / {img2.numel()}")
    pad = window_size // 2
    img1_padded = F.pad(img1, (pad, pad, pad, pad, pad, pad), mode='reflect')
    img2_padded = F.pad(img2, (pad, pad, pad, pad, pad, pad), mode='reflect')
    mu1 = F.conv3d(img1_padded, window, padding = 0, groups = channel)
    mu2 = F.conv3d(img2_padded, window, padding = 0, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1_padded*img1_padded, window, padding = 0, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2_padded*img2_padded, window, padding = 0, groups = channel) - mu2_sq
    sigma1_sq = torch.clamp(sigma1_sq, min=0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0)
    sigma12 = F.conv3d(img1_padded*img2_padded, window, padding = 0, groups = channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    # print(f"mu1: {mu1}, mu2: {mu2}, sigma1_sq: {sigma1_sq}, sigma2_sq: {sigma2_sq}, sigma12: {sigma12}")

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    print(f"ssim_map data range: {ssim_map.min()} to {ssim_map.max()}")
    print(f"ssim_map mean: {ssim_map.mean()}")
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(dim=[1,2,3,4])
    


# class SSIM(torch.nn.Module):
#     def __init__(self, window_size = 11, size_average = True):
#         super(SSIM, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         window = create_window(window_size, self.channel)
#         self.register_buffer('window', window)

#     def forward(self, img1, img2):
#         img1f = img1.float()
#         img2f = img2.float()
#         (_, channel, _, _) = img1f.size()

#         need_new = (
#             channel != self.channel
#             or self.window.device != img1f.device
#             or self.window.dtype != img1f.dtype
#         )

#         if need_new:
#             window = create_window(self.window_size, channel).to(device=img1f.device, dtype=img1f.dtype)
#             self.window.set_(window)
#             self.channel = channel
#         else:
#             window = self.window

#         return _ssim(img1f, img2f, window, self.window_size, channel, self.size_average)
    
    
class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        window = create_window_3D(window_size, self.channel)
        self.register_buffer('window', window) # 这里注册完window之后已经就是self.window了

    def forward(self, img1, img2):
        img1f = img1.float()
        img2f = img2.float()
        (_, channel, _, _, _) = img1f.size()

        need_new = (
            channel != self.channel
            or self.window.device != img1f.device
            or self.window.dtype != img1f.dtype
        )

        if need_new:
            window = create_window_3D(self.window_size, channel).to(device=img1f.device, dtype=img1f.dtype)
            self.window = window
            self.channel = channel
        else:
            window = self.window

        return _ssim_3D(img1f, img2f, window, self.window_size, channel, self.size_average)

    
def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.to(img1.device)
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.to(img1.device)
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)


# ----------------------------
# 3. Gradient Loss
# ----------------------------

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 计算 x, y, z 方向的梯度
        grad_pred_x = pred[..., 1:, :, :] - pred[..., :-1, :, :]
        grad_pred_y = pred[..., :, 1:, :] - pred[..., :, :-1, :]
        grad_pred_z = pred[..., :, :, 1:] - pred[..., :, :, :-1]

        grad_target_x = target[..., 1:, :, :] - target[..., :-1, :, :]
        grad_target_y = target[..., :, 1:, :] - target[..., :, :-1, :]
        grad_target_z = target[..., :, :, 1:] - target[..., :, :, :-1]

        loss_x = F.l1_loss(grad_pred_x, grad_target_x)
        loss_y = F.l1_loss(grad_pred_y, grad_target_y)
        loss_z = F.l1_loss(grad_pred_z, grad_target_z)

        return (loss_x + loss_y + loss_z) / 3.0



class HybridLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 1.0,
        l2_weight: float = 1.0,
        ssim_weight: float = 0.5,
        grad_weight: float = 0.1,
        perceptual_weight: float = 0.01,
        logger = None,
    ):
        super().__init__()
        self.l1_loss = L1Loss()
        # self.my_ssim_loss = SSIM3D()
        self.ssim_loss = SSIMLoss(spatial_dims=3,data_range=1.0)
        self.grad_loss = GradientLoss()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight
        self.perceptual_weight = perceptual_weight
        self.logger = logger
        self.current_epoch = 0
        self.l2_loss = MSELoss()
        self.l2_weight = l2_weight

        # self.perceptual_loss = PerceptualLoss(
        #     spatial_dims=3,
        #     network_type="medicalnet_resnet50_23datasets",
        #     is_fake_3d=False,
        #     channel_wise=False,
        # )
        self.perceptual_loss = MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.l1_weight > 0:
            self.l1 = self.l1_loss(pred, target)
        else:
            self.l1 = torch.tensor(0.0, device=pred.device)

        if self.ssim_weight > 0:
            # self.ssim = self.ssim_loss(pred, target)
            pred_ssim = pred.unsqueeze(1)  # [B, D, H, W]
            target_ssim = target.unsqueeze(1)
            self.ssim = 1.0 - ms_ssim(pred_ssim, target_ssim, data_range=1.0, size_average=False)
        else:
            self.ssim = torch.tensor(0.0, device=pred.device)

        if self.grad_weight > 0:        
            self.grad = self.grad_loss(pred, target)
        else:
            self.grad = torch.tensor(0.0, device=pred.device)

        if self.perceptual_weight > 0:
            self.perceptual = self.perceptual_loss(pred, target)
        else:
            self.perceptual = torch.tensor(0.0, device=pred.device)

        if self.l2_weight > 0:
            self.l2 = self.l2_loss(pred, target)
        else:
            self.l2 = torch.tensor(0.0, device=pred.device)
        
        if self.logger:
            self.logger.info(f"L1 Loss: {self.l1.item():.4f}, L2 Loss: {self.l2.item():.4f}, "
                             f"SSIM Loss: {self.ssim.item():.4f}, Gradient Loss: {self.grad.item():.4f}, "
                             f"Perceptual Loss: {self.perceptual.item():.4f}"
                )

        self.total_loss = self.l1_weight * self.l1 + self.l2_weight * self.l2 + self.ssim_weight * self.ssim + self.grad_weight * self.grad + self.perceptual_weight * self.perceptual
        # self.total_loss = self.l1_weight * self.l1 + self.ssim_weight * self.ssim + self.grad_weight * self.grad 

        

        return self.total_loss