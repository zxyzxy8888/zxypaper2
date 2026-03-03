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

CLASS_TO_ID = {"AD": 0, "MCI": 1, "CN": 2}


def _normalize_research_group(label):
    """Convert string label like 'AD' to an integer id used for class guidance."""
    key = str(label).strip().upper()
    if key not in CLASS_TO_ID:
        raise ValueError(f"Unknown class '{label}'. Expected one of {list(CLASS_TO_ID.keys())}.")
    return CLASS_TO_ID[key]

class MRIPETDataset(Dataset):
    """
    Docstring for MRIPETDataset
    Dataset 的职责是加载单个样本(或者成对样本)，不是管理数据划分
    """
    def __init__(self, paired_files, pad_multiple=8, target_shape=None):
        self.paired_files = paired_files
        self.pad_multiple = pad_multiple
        self.target_shape = target_shape

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        mri_path, pet_path, research_group = self.paired_files[idx]
        
        # 加载
        mri = nib.load(mri_path).get_fdata().astype(np.float32)
        pet = nib.load(pet_path).get_fdata().astype(np.float32)
        mri = mri[3:179, 5:213, 0:176]
        pet = pet[3:179, 5:213, 0:176]
        
        # 转为 tensor 并加 batch/channel 维度 -> (1, 1, D, H, W)
        mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        pet = torch.from_numpy(pet).unsqueeze(0).unsqueeze(0)

        # # 4. 处理尺寸：下采样 OR Padding
        # if self.target_shape is not None:
        #     # 方案 A: 强制下采样 (针对 5090 显存优化)
        #     # mode='trilinear' 是 3D 图像的标准插值方法
        #     # align_corners=False 是默认推荐配置
        #     mri = F.interpolate(mri, size=self.target_shape, mode='trilinear', align_corners=False)
        #     pet = F.interpolate(pet, size=self.target_shape, mode='trilinear', align_corners=False)
        # # elif self.pad_multiple > 1:
        #     # 方案 B: 原始 Padding 逻辑 (保持分辨率)
        #     mri, _ = pad_to_multiple_of(mri, self.pad_multiple)
        #     pet, _ = pad_to_multiple_of(pet, self.pad_multiple)

        # # Padding 到 8 的倍数
        # if self.pad_multiple > 1:
        #     mri, orig_size = pad_to_multiple_of(mri, self.pad_multiple)
        #     pet, _ = pad_to_multiple_of(pet, self.pad_multiple)
        #     # 注意：我们只保存 mri 的 orig_size（两者相同）
        #     # 把 orig_size 存下来，用于训练后裁剪（可选）
        #     # 这里先不返回，除非你需要测试时精确还原
        # else:
        #     orig_size = mri.shape[2:]

        # 移除多余的 batch 维度（DataLoader 会加回来）
        mri = mri.squeeze(0)  # (1, D, H, W)
        pet = pet.squeeze(0)  # (1, D, H, W)
        
        # # Debug: Check after all preprocessing
        # print(f"After preprocessing - MRI: {mri.shape}, range: [{mri.min():.6f}, {mri.max():.6f}]")
        # print(f"After preprocessing - PET: {pet.shape}, range: [{pet.min():.6f}, {pet.max():.6f}]")
        # print(f"After preprocessing - PET non-zero: {torch.count_nonzero(pet).item()} / {pet.numel()}\n")

        class_id = torch.tensor(_normalize_research_group(research_group), dtype=torch.long)

        return mri, pet, class_id, mri_path, pet_path
    
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
