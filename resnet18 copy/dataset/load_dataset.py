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
                     train_ratio=0.7, val_ratio=0.15, random_state=2):
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
label_map = {'MCI': 0, 'AD': 1}  # 01必须是健康到患病

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