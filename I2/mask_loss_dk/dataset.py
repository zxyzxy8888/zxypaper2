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

"""
Docstring for zxy.paper.zxypaper.code.components.dataset
purpose: 负责加载MRI和PET数据集的数据集类 (适配 182x218x182 输入)
Date: 2026-02-07
Author: Zhiyuan Liu
"""

import os
import nibabel as nib
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class MRIPETDataset(Dataset):
    def __init__(self, paired_files, mask_dir, pad_multiple=8, target_shape=(128, 128, 128)):
        self.mask_dir = mask_dir
        self.pad_multiple = pad_multiple
        self.target_shape = target_shape
        
        # --- 1. 初始化时检查 Mask 完整性 ---
        self.paired_files = [] 
        skipped_count = 0
        
        print(f"正在检查 {len(paired_files)} 个样本的 Mask 文件...")
        
        for pair in paired_files:
            mri_path = pair[0]
            # 构造 Mask 文件名
            mri_filename = os.path.basename(mri_path)
            if mri_filename.endswith('.nii.gz'):
                base_name = mri_filename[:-7]
            elif mri_filename.endswith('.nii'):
                base_name = mri_filename[:-4]
            else:
                base_name = mri_filename
                
            # 根据你的文件名规则拼接
            mask_name = base_name + "dk_atlas_in_mni.nii.gz"
            mask_path = os.path.join(self.mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.paired_files.append(pair)
            else:
                skipped_count += 1
        
        print(f"数据集准备完毕: 有效样本 {len(self.paired_files)} 个 (剔除 {skipped_count} 个缺失Mask的样本)")

    def __len__(self):
        return len(self.paired_files)

    def __getitem__(self, idx):
        mri_path, pet_path, research_group = self.paired_files[idx]
        
        # 重新构造 Mask 路径
        mri_filename = os.path.basename(mri_path)
        if mri_filename.endswith('.nii.gz'):
            base_name = mri_filename[:-7]
        elif mri_filename.endswith('.nii'):
            base_name = mri_filename[:-4]
        else:
            base_name = mri_filename
        mask_name = base_name + "dk_atlas_in_mni.nii.gz"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 1. 加载数据
        try:
            mri = nib.load(mri_path).get_fdata().astype(np.float32)
            pet = nib.load(pet_path).get_fdata().astype(np.float32)
            mask = nib.load(mask_path).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"读取错误: {mri_path}")
            raise e

        # 2. 统一裁剪 (处理 182x218x182 的标准 MNI 输入)
        # 裁剪范围: x[3:179], y[5:213], z[0:176] -> 结果大小: 176x208x176
        # 这个裁剪是专门针对 MNI 182 尺寸设计的，去除了周围的黑边
        
        if mri.shape == (182, 218, 182):
            mri = mri[3:179, 5:213, 0:176]
            pet = pet[3:179, 5:213, 0:176]
            mask = mask[3:179, 5:213, 0:176] # Mask 必须做完全一样的裁剪！
        
        # 转 Tensor: (D, H, W) -> (1, 1, D, H, W) 用于 interpolate
        mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)
        pet = torch.from_numpy(pet).unsqueeze(0).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)

        # 3. 尺寸缩放 (Resize) 到目标大小 (如 128x128x128)
        if self.target_shape is not None:
            # MRI 和 PET: 使用三线性插值 (Trilinear) -> 保持平滑
            mri = F.interpolate(mri, size=self.target_shape, mode='trilinear', align_corners=False)
            pet = F.interpolate(pet, size=self.target_shape, mode='trilinear', align_corners=False)
            
            # Mask: 必须使用最近邻插值 (Nearest) -> 保持标签整数值 (17还是17，不会变成17.5)
            mask = F.interpolate(mask, size=self.target_shape, mode='nearest')

        # 压缩维度返回: (1, D, H, W)
        mri = mri.squeeze(0)
        pet = pet.squeeze(0)
        mask = mask.squeeze(0)

        return mri, pet, mask, research_group, mri_path
    
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
