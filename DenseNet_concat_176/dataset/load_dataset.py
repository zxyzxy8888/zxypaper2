import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def load_pairing_csv(csv_path):
    """
    读取配对 CSV 文件，返回 List[Tuple[str, str, str]]: [(mri_name, pet_name, research_group), ...]
    """
    df = pd.read_csv(csv_path)
    # 去除空行
    df = df.dropna()
    # 构建列表（而不是字典）
    pairing = []
    for _, row in df.iterrows():
        mri_name = row['MRI'].strip()
        pet_name = row['PET'].strip()
        research_group = row['Research Group'].strip()
        pairing.append((mri_name, pet_name, research_group))

    return pairing
def extract_subject_id(filename):
    """
    从文件名提取完整 subject ID
      "MRI_137_S_1041_I56100.nii.gz" → "137_S_1041"、
    Returns:
        str: subject ID like "137_S_1041"
    """
    match = re.search(r"_(\d+_S_\d+)_", filename)
    if match:
        return match.group(1)
    else:
        raise ValueError(f"Cannot extract subject ID from filename: {filename}")


def get_paired_files_with_subjects(mri_dir, pet_dir, csv_path):
    """
    根据 CSV 和目录构建配对，并按完整 Subject ID 分组。

    Returns:
        paired_files: List[Tuple[str, str, str]] — (mri_path, pet_path, research_group)
        subject_to_files: Dict[str, List[Tuple]] — {subject_id: [(mri, pet, group), ...]}
    """
    pairing = load_pairing_csv(csv_path)
    # 获取目录中所有 .nii.gz 文件（仅文件名）
    mri_files_in_dir = set(os.listdir(mri_dir))
    pet_files_in_dir = set(os.listdir(pet_dir))
    paired_files = []
    subject_to_files = {}
    for mri_name, pet_name, research_group in pairing:
        # 检查文件是否存在
        if mri_name not in mri_files_in_dir:
            print(f"⚠️ Warning: MRI file not found: {mri_name}")
            continue
        if pet_name not in pet_files_in_dir:
            print(f"⚠️ Warning: PET file not found: {pet_name}")
            continue
        mri_path = os.path.join(mri_dir, mri_name)
        pet_path = os.path.join(pet_dir, pet_name)
        # 提取 Subject ID（关键！）
        subject_id = extract_subject_id(mri_name)  # e.g., "137_S_1041"
        # 添加到列表
        paired_files.append((mri_path, pet_path, research_group))
        # 按 subject 分组
        if subject_id not in subject_to_files:
            subject_to_files[subject_id] = []
        subject_to_files[subject_id].append((mri_path, pet_path, research_group))
    print(f"✅ Loaded {len(paired_files)} paired scans from {len(subject_to_files)} unique subjects.")
    return paired_files, subject_to_files


def split_by_subject(paired_files, subject_to_files,
                     train_ratio=0.7, val_ratio=0.15, random_state=562):
    """
    按完整 Subject ID 划分，确保同一 subject 的所有 scan 在同一集合。
    """
    subject_ids = list(subject_to_files.keys())

    # 第一次划分：train vs (val+test)
    train_subs, temp_subs = train_test_split(
        subject_ids, test_size=(1 - train_ratio), random_state=random_state
    )
    # 第二次划分：val vs test
    val_subs, test_subs = train_test_split(
        temp_subs, test_size=val_ratio / (1 - train_ratio), random_state=random_state
    )
    def _filter_by_subjects(files_list, subject_set):
        subject_set = set(subject_set)
        return [f for f in files_list if extract_subject_id(os.path.basename(f[0])) in subject_set]
    train_files = _filter_by_subjects(paired_files, train_subs)
    val_files = _filter_by_subjects(paired_files, val_subs)
    test_files = _filter_by_subjects(paired_files, test_subs)
    # 打印统计
    print(f"📊 Train: {len(train_files)} scans ({len(train_subs)} subjects)")
    print(f"📊 Val:   {len(val_files)} scans ({len(val_subs)} subjects)")
    print(f"📊 Test:  {len(test_files)} scans ({len(test_subs)} subjects)")
    print(f"test subjects: {test_subs}")
    return train_files, val_files, test_files
# 标签映射
label_map = {'CN': 0, 'AD': 1}  # 01必须是健康到患病

class MRIPETDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        file_list: List of tuples (mri_path, pet_path, label)
        """
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        mri_path, pet_path, label = self.file_list[idx]
        # 加载 MRI 和 PET 图像
        mri_img = nib.load(mri_path).get_fdata()
        pet_img = nib.load(pet_path).get_fdata()
        # 转换为 tensor，并添加通道维度 [C, D, H, W]
        mri_tensor = torch.tensor(mri_img, dtype=torch.float32).unsqueeze(0)
        pet_tensor = torch.tensor(pet_img, dtype=torch.float32).unsqueeze(0)
        # 将 MRI 和 PET 合并为一个多通道输入：[2, D, H, W]
        input_tensor = torch.cat([mri_tensor, pet_tensor], dim=0)
        # 标签转换为整数
        label_tensor = torch.tensor(label_map[label], dtype=torch.long)
        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, label_tensor
# """
# Docstring for zxy.paper.zxypaper.code.components.dataset
# purpose: 负责加载MRI和PET数据集的数据集类
# Date: 2025-12-11
# Author: Zhiyuan Liu
# """

# import os
# import nibabel as nib
# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import torch.nn.functional as F

# class MRIPETDataset(Dataset):
#     """
#     Docstring for MRIPETDataset
#     Dataset 的职责是加载单个样本(或者成对样本)，不是管理数据划分
#     """
#     def __init__(self, paired_files, pad_multiple=8):
#         self.paired_files = paired_files
#         self.pad_multiple = pad_multiple

#     def __len__(self):
#         return len(self.paired_files)

#     def __getitem__(self, idx):
#         mri_path, pet_path, research_group = self.paired_files[idx]
        
#         # 加载
#         mri = nib.load(mri_path).get_fdata().astype(np.float32)
#         pet = nib.load(pet_path).get_fdata().astype(np.float32)

#         # bbox = get_nonzero_bounding_box(mri, threshold=0)
#         # print_bounding_box_info(bbox, name="MRI")
#         # bbox_pet = get_nonzero_bounding_box(pet, threshold=0)
#         # print_bounding_box_info(bbox_pet, name="PET")

#         # 经过上面注释的代码，可以看出MRI和PET外围都是有大量0存在的
#         # 在这里我们对两者都进行裁剪，保留非零区域，为了后面的padding
#         # 也就是Unet encoder和decoder之间传递信息的维度是相同的，也就是说都是2的depth次幂的倍数
#         # 原本是182x218x182，经过裁剪变为176*208*176
#         # 本数据大概是这样的范围：
#         # 原始shape:      (182, 218, 182)
#         # 有效shape:      (148, 184, 156)
#         # X轴范围: [ 16, 164)
#         # Y轴范围: [ 18, 202)
#         # Z轴范围: [  0, 156)
#         mri = mri[3:179, 5:213, 0:176]
#         pet = pet[3:179, 5:213, 0:176]
        

#         # Debug: Check if PET data is all zeros
#         # print(f"\n=== Dataset Debug Info (Sample {idx}) ===")
#         # print(f"idx: {idx}")
#         # print(f"MRI path: {mri_path}")
#         # print(f"PET path: {pet_path}")
#         # print(f"MRI shape: {mri.shape}, range: [{mri.min():.6f}, {mri.max():.6f}]")
#         # print(f"PET shape: {pet.shape}, range: [{pet.min():.6f}, {pet.max():.6f}]")
#         # print(f"MRI non-zero count: {np.count_nonzero(mri)} / {mri.size}")
#         # print(f"PET non-zero count: {np.count_nonzero(pet)} / {pet.size}")
#         # print("="*50 + "\n")
        
#         # 转为 tensor 并加 batch/channel 维度 -> (1, 1, D, H, W)
#         mri = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
#         pet = torch.from_numpy(pet).unsqueeze(0).unsqueeze(0)

#         # Padding 到 8 的倍数
#         if self.pad_multiple > 1:
#             mri, orig_size = pad_to_multiple_of(mri, self.pad_multiple)
#             pet, _ = pad_to_multiple_of(pet, self.pad_multiple)
#             # 注意：我们只保存 mri 的 orig_size（两者相同）
#             # 把 orig_size 存下来，用于训练后裁剪（可选）
#             # 这里先不返回，除非你需要测试时精确还原
#         else:
#             orig_size = mri.shape[2:]

#         # 移除多余的 batch 维度（DataLoader 会加回来）
#         mri = mri.squeeze(0)  # (1, D, H, W)
#         pet = pet.squeeze(0)  # (1, D, H, W)
#         input_tensor = torch.cat([mri, pet], dim=0)  # (2, D, H, W)
#         label_tensor = torch.tensor(label_map[research_group], dtype=torch.long)
#         # # Debug: Check after all preprocessing
#         # print(f"After preprocessing - MRI: {mri.shape}, range: [{mri.min():.6f}, {mri.max():.6f}]")
#         # print(f"After preprocessing - PET: {pet.shape}, range: [{pet.min():.6f}, {pet.max():.6f}]")
#         # print(f"After preprocessing - PET non-zero: {torch.count_nonzero(pet).item()} / {pet.numel()}\n")

#         return input_tensor, label_tensor
    
# def get_nonzero_bounding_box(data, threshold=0):
#     """
#     找到3D数据中非零（或大于threshold）区域的边界框
    
#     Args:
#         data: numpy array or torch tensor, shape (D, H, W) or (H, W, D)
#         threshold: 阈值，默认为0，即找非零区域
    
#     Returns:
#         dict: {
#             'x_range': (x_start, x_end),
#             'y_range': (y_start, y_end), 
#             'z_range': (z_start, z_end),
#             'valid_shape': (x_size, y_size, z_size),
#             'original_shape': (x, y, z)
#         }
    
#     Example:
#         >>> mri = np.random.rand(182, 218, 182)
#         >>> mri[175:, :, :] = 0  # 后面几层全是0
#         >>> bbox = get_nonzero_bounding_box(mri)
#         >>> print(f"有效数据范围: {bbox['valid_shape']}")
#     """
#     if torch.is_tensor(data):
#         data = data.cpu().numpy()
    
#     # 找到每个维度上非零的索引
#     nonzero_mask = data > threshold
    
#     # 获取原始shape
#     original_shape = data.shape
    
#     # 沿着每个轴找非零区域
#     nonzero_indices = np.where(nonzero_mask)
    
#     if len(nonzero_indices[0]) == 0:
#         # 全是0的情况
#         return {
#             'x_range': (0, 0),
#             'y_range': (0, 0),
#             'z_range': (0, 0),
#             'valid_shape': (0, 0, 0),
#             'original_shape': original_shape
#         }
    
#     # 找到每个维度的最小和最大索引
#     x_min, x_max = nonzero_indices[0].min(), nonzero_indices[0].max() + 1
#     y_min, y_max = nonzero_indices[1].min(), nonzero_indices[1].max() + 1
#     z_min, z_max = nonzero_indices[2].min(), nonzero_indices[2].max() + 1
    
#     return {
#         'x_range': (int(x_min), int(x_max)),
#         'y_range': (int(y_min), int(y_max)),
#         'z_range': (int(z_min), int(z_max)),
#         'valid_shape': (int(x_max - x_min), int(y_max - y_min), int(z_max - z_min)),
#         'original_shape': original_shape
#     }


# def print_bounding_box_info(bbox_info, name="Data"):
#     """
#     打印边界框信息
    
#     Args:
#         bbox_info: get_nonzero_bounding_box返回的字典
#         name: 数据名称（如"MRI"或"PET"）
#     """
#     print(f"\n{'='*50}")
#     print(f"{name} Bounding Box Information")
#     print(f"{'='*50}")
#     print(f"原始shape:      {bbox_info['original_shape']}")
#     print(f"有效shape:      {bbox_info['valid_shape']}")
#     print(f"X轴范围: [{bbox_info['x_range'][0]:3d}, {bbox_info['x_range'][1]:3d})")
#     print(f"Y轴范围: [{bbox_info['y_range'][0]:3d}, {bbox_info['y_range'][1]:3d})")
#     print(f"Z轴范围: [{bbox_info['z_range'][0]:3d}, {bbox_info['z_range'][1]:3d})")
    
#     # 计算padding大小
#     x_pad_before = bbox_info['x_range'][0]
#     x_pad_after = bbox_info['original_shape'][0] - bbox_info['x_range'][1]
#     y_pad_before = bbox_info['y_range'][0]
#     y_pad_after = bbox_info['original_shape'][1] - bbox_info['y_range'][1]
#     z_pad_before = bbox_info['z_range'][0]
#     z_pad_after = bbox_info['original_shape'][2] - bbox_info['z_range'][1]
    
#     print(f"\nPadding信息:")
#     print(f"X轴: 前{x_pad_before}个, 后{x_pad_after}个")
#     print(f"Y轴: 前{y_pad_before}个, 后{y_pad_after}个")
#     print(f"Z轴: 前{z_pad_before}个, 后{z_pad_after}个")
#     print(f"{'='*50}\n")


# def pad_to_multiple_of(tensor, multiple=8):
#     """
#     将 5D 张量 (B, C, D, H, W) 的空间维度 pad 到 multiple 的倍数
#     返回 padded tensor 和原始尺寸（用于后续裁剪）
#     """
#     _, _, d, h, w = tensor.shape
#     new_d = ((d - 1) // multiple + 1) * multiple
#     new_h = ((h - 1) // multiple + 1) * multiple
#     new_w = ((w - 1) // multiple + 1) * multiple

#     pad_d = new_d - d
#     pad_h = new_h - h
#     pad_w = new_w - w

#     # F.pad 的顺序是 (W前, W后, H前, H后, D前, D后)
#     padded = F.pad(tensor, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
#     return padded, (d, h, w)