# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Modified for 3D MRI-to-PET Generation (Training from Scratch)
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import importlib.util
from pathlib import Path
from UNet_linearatt import UNetModel
# except Exception:
#     local_unet_hyphen = Path(__file__).with_name("U-Net.py")
#     if local_unet_hyphen.exists():
#         spec = importlib.util.spec_from_file_location("i2_i2sb_local_unet", str(local_unet_hyphen))
#         module = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(module)
#         UNetModel = module.UNetModel
#     else:
#         from I2SB.unet import UNetModel

try:
    from . import util
except Exception:
    import util

class Image3DNet(torch.nn.Module):
    def __init__(self, log, noise_levels, use_fp16=False, cond=True, image_size=96, num_classes=3):
        """
        :param log: 日志记录器
        :param noise_levels: 噪声水平表 (Buffer)
        :param use_fp16: 是否使用半精度
        :param cond: 是否为条件生成 (MRI -> PET 通常为 True)
        :param image_size: 3D 图像的空间尺寸 (建议 128 或 96)
        """
        super(Image3DNet, self).__init__()

        # ----------------------------------------------------------------
        # 1. 定义 3D U-Net 的超参数 (不再从 pickle 读取)
        # ----------------------------------------------------------------
        # 如果是条件生成，输入通道 = Noisy PET (1) + MRI (1) = 2
        # 如果是无条件，输入通道 = Noisy PET (1)
        in_channels = 2 if cond else 1
        out_channels = 1  # 输出只有 PET (1通道)

        model_kwargs = dict(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=64,            # 基础通道数，显存不够可改为 32
            out_channels=out_channels,
            num_res_blocks=2,             # 每个层级的残差块数量
            attention_resolutions=(12),   # 仅在极低分辨率(如8x8x8)使用Attention，防OOM
            dropout=0.,
            channel_mult=(1, 2, 4, 8),    # 通道倍增: 64 -> 128 -> 256 -> 512
            conv_resample=True,
            dims=3,                       # <--- 关键：开启 3D 模式
            num_classes=num_classes,      # 通过标签进行类别引导
            use_checkpoint=True,          # <--- 关键：3D 训练强烈建议开启梯度检查点
            use_fp16=use_fp16,
            num_heads=4,
            use_scale_shift_norm=True,
            resblock_updown=False,
        )

        # ----------------------------------------------------------------
        # 2. 初始化模型 (随机初始化权重)
        # ----------------------------------------------------------------
        self.diffusion_model = UNetModel(**model_kwargs)
        
        # 打印模型参数量
        log.info(f"[Net] Initialized 3D UNet from scratch! "
                 f"Input Channels={in_channels}, Dims=3. "
                 f"Size={util.count_parameters(self.diffusion_model)} parameters.")
        self.cond = cond
        self.noise_levels = noise_levels
        self.num_classes = num_classes

    def forward(self, x, steps, cond=None, y=None):
        """
        :param x: 当前的 Noisy PET 图像 [B, 1, D, H, W]
        :param steps: 当前时间步索引 [B]
        :param cond: 条件图像 (MRI) [B, 1, D, H, W]
        """
        # 获取时间嵌入
        t = self.noise_levels[steps].detach()
        assert t.dim() == 1 and t.shape[0] == x.shape[0]

        # 拼接条件 (MRI)
        if self.cond:
            assert cond is not None, "Model is conditional but no condition provided!"
            # 在通道维度 (dim=1) 拼接: [B, 1, ...] + [B, 1, ...] -> [B, 2, ...]
            x = torch.cat([x, cond], dim=1)
        
        # 前向传播
        return self.diffusion_model(x, t, y=y)


class Image256Net(Image3DNet):
    pass