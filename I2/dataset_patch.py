"""
Docstring for zxy.paper.zxypaper.code.components.dataset
purpose: 负责加载MRI和PET数据集的数据集类
Date: 2025-12-11
Author: Zhiyuan Liu
"""

import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import random

class MRIPETDataset(Dataset):
    """
    改进版：支持 3D Patch-based 训练的 MRIPETDataset
    """
    def __init__(self, paired_files, patch_size=(96, 96, 96), is_train=True, threshold=0.1):
        """
        Args:
            paired_files: 配对文件列表
            patch_size: 裁剪的 Patch 大小，例如 (96, 96, 96)
            is_train: 是否为训练模式。训练模式下使用随机裁剪，测试模式可返回整脑或固定中心裁剪
            threshold: 过滤阈值。要求 Patch 中脑部像素（MRI > 0）占比至少达到该比例，否则重新采样
        """
        self.paired_files = paired_files
        self.patch_size = patch_size
        self.is_train = is_train
        self.threshold = threshold

    def __len__(self):
        return len(self.paired_files)

    def _random_crop_3d(self, mri, pet, patch_size):
        """
        执行空间同步的随机裁剪
        """
        d, h, w = mri.shape
        pd, ph, pw = patch_size

        # 最大重试次数，防止在某些空区域死循环
        for _ in range(20):
            # 随机选择起始坐标
            z = random.randint(0, d - pd)
            y = random.randint(0, h - ph)
            x = random.randint(0, w - pw)

            mri_patch = mri[z:z+pd, y:y+ph, x:x+pw]
            pet_patch = pet[z:z+pd, y:y+ph, x:x+pw]

            # 过滤逻辑：利用 MRI > 0 (脑区) 判断该 Patch 是否有效
            # 如果脑组织占比太低（比如切到了空气），则重新寻找坐标
            brain_ratio = np.sum(mri_patch > 0) / mri_patch.size
            if brain_ratio > self.threshold:
                return mri_patch, pet_patch
        
        # 如果20次都没找到合适的，就返回最后一次的结果（保底）
        return mri_patch, pet_patch

    def __getitem__(self, idx):
        mri_path, pet_path, research_group = self.paired_files[idx]
        
        # 1. 加载原始分辨率数据 (不进行插值)
        mri_nib = nib.load(mri_path)
        pet_nib = nib.load(pet_path)
        mri = mri_nib.get_fdata().astype(np.float32)
        pet = pet_nib.get_fdata().astype(np.float32)

        # 2. 基础预裁剪 (去除外围绝对多余的零值空间，减少采样压力)
        # 你之前的范围是 [3:179, 5:213, 0:176]，可以保留
        mri = mri[3:179, 5:213, 0:176]
        pet = pet[3:179, 5:213, 0:176]

        # 3. 执行 Patch 裁剪
        if self.is_train and self.patch_size is not None:
            # 训练模式：随机同步裁剪
            mri_data, pet_data = self._random_crop_3d(mri, pet, self.patch_size)
        else:
        # 测试/验证模式：返回【完整】的 3D 图像
        # 不要在这里 resize 或 crop，滑动窗口会处理它
        mri_data = mri
        pet_data = pet

        # 4. 转为 tensor -> (1, D, H, W)
        mri_tensor = torch.from_numpy(mri_data).unsqueeze(0)
        pet_tensor = torch.from_numpy(pet_data).unsqueeze(0)

        return mri_tensor, pet_tensor, research_group, mri_path, pet_path
    
def get_nonzero_bounding_box(data, threshold=0):
    """
    找到3D数据中非零（或大于threshold）区域的边界框
    
    Args:
        data: numpy array or torch tensor, shape (D, H, W) or (H, W, D)
        threshold: 阈值，默认为0，即找非零区域
    
    Returns:
        dict: {
            'x_range': (x_start, x_end),
            'y_range': (y_start, y_end), 
            'z_range': (z_start, z_end),
            'valid_shape': (x_size, y_size, z_size),
            'original_shape': (x, y, z)
        }
    
    Example:
        >>> mri = np.random.rand(182, 218, 182)
        >>> mri[175:, :, :] = 0  # 后面几层全是0
        >>> bbox = get_nonzero_bounding_box(mri)
        >>> print(f"有效数据范围: {bbox['valid_shape']}")
    """
    if torch.is_tensor(data):
        data = data.cpu().numpy()
    
    # 找到每个维度上非零的索引
    nonzero_mask = data > threshold
    
    # 获取原始shape
    original_shape = data.shape
    
    # 沿着每个轴找非零区域
    nonzero_indices = np.where(nonzero_mask)
    
    if len(nonzero_indices[0]) == 0:
        # 全是0的情况
        return {
            'x_range': (0, 0),
            'y_range': (0, 0),
            'z_range': (0, 0),
            'valid_shape': (0, 0, 0),
            'original_shape': original_shape
        }
    
    # 找到每个维度的最小和最大索引
    x_min, x_max = nonzero_indices[0].min(), nonzero_indices[0].max() + 1
    y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max() + 1
    z_min, z_max = nonzero_indices[2].min(), nonzero_indices[2].max() + 1
    
    return {
        'x_range': (int(x_min), int(x_max)),
        'y_range': (int(y_min), int(y_max)),
        'z_range': (int(z_min), int(z_max)),
        'valid_shape': (int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)),
        'original_shape': original_shape
    }


def print_bounding_box_info(bbox_info, name="Data"):
    """
    打印边界框信息
    
    Args:
        bbox_info: get_nonzero_bounding_box返回的字典
        name: 数据名称（如"MRI"或"PET"）
    """
    print(f"\n{'='*50}")
    print(f"{name} Bounding Box Information")
    print(f"{'='*50}")
    print(f"原始shape:      {bbox_info['original_shape']}")
    print(f"有效shape:      {bbox_info['valid_shape']}")
    print(f"X轴范围: [{bbox_info['x_range'][0]:3d}, {bbox_info['x_range'][1]:3d})")
    print(f"Y轴范围: [{bbox_info['y_range'][0]:3d}, {bbox_info['y_range'][1]:3d})")
    print(f"Z轴范围: [{bbox_info['z_range'][0]:3d}, {bbox_info['z_range'][1]:3d})")
    
    # 计算padding大小
    x_pad_before = bbox_info['x_range'][0]
    x_pad_after = bbox_info['original_shape'][0] - bbox_info['x_range'][1]
    y_pad_before = bbox_info['y_range'][0]
    y_pad_after = bbox_info['original_shape'][1] - bbox_info['y_range'][1]
    z_pad_before = bbox_info['z_range'][0]
    z_pad_after = bbox_info['original_shape'][2] - bbox_info['z_range'][1]
    
    print(f"\nPadding信息:")
    print(f"X轴: 前{x_pad_before}个, 后{x_pad_after}个")
    print(f"Y轴: 前{y_pad_before}个, 后{y_pad_after}个")
    print(f"Z轴: 前{z_pad_before}个, 后{z_pad_after}个")
    print(f"{'='*50}\n")


def pad_to_multiple_of(tensor, multiple=8):
    """
    将 5D 张量 (B, C, D, H, W) 的空间维度 pad 到 multiple 的倍数
    返回 padded tensor 和原始尺寸（用于后续裁剪）
    """
    _, _, d, h, w = tensor.shape
    new_d = ((d - 1) // multiple + 1) * multiple
    new_h = ((h - 1) // multiple + 1) * multiple
    new_w = ((w - 1) // multiple + 1) * multiple

    pad_d = new_d - d
    pad_h = new_h - h
    pad_w = new_w - w

    # F.pad 的顺序是 (W前, W后, H前, H后, D前, D后)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
    return padded, (d, h, w)
