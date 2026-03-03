import pandas as pd
import os
from pathlib import Path
import re
from sklearn.model_selection import train_test_split


def load_pairing_csv(csv_path):
    """
    读取配对 CSV 文件，返回 List[Tuple[str, str, str]]: [(mri_name, pet_name, class_label), ...]
    兼容列名：优先匹配 MRI/PET/Research Group 等常见写法，否则使用前三列。
    """
    df = pd.read_csv(csv_path)

    # 去除空行
    df = df.dropna()

    def _pick_column(df_obj, candidates, fallback_idx):
        cols_lower = [c.lower().strip() for c in df_obj.columns]
        for cand in candidates:
            if cand.lower() in cols_lower:
                return df_obj.columns[cols_lower.index(cand.lower())]
        if len(df_obj.columns) > fallback_idx:
            return df_obj.columns[fallback_idx]
        raise ValueError(f"CSV 缺少列：{candidates}")

    mri_col = _pick_column(df, ["mri", "mri_file", "mri filename"], 0)
    pet_col = _pick_column(df, ["pet", "pet_file", "pet filename"], 1)
    label_col = _pick_column(df, ["research group", "group", "label", "class"], 2)

    # 构建列表（而不是字典）
    pairing = []
    for _, row in df.iterrows():
        mri_name = str(row[mri_col]).strip()
        pet_name = str(row[pet_col]).strip()
        research_group = str(row[label_col]).strip()
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
                     train_ratio=0.7, val_ratio=0.15, random_state=42):
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
    val_files   = _filter_by_subjects(paired_files, val_subs)
    test_files  = _filter_by_subjects(paired_files, test_subs)

    # 打印统计
    print(f"📊 Train: {len(train_files)} scans ({len(train_subs)} subjects)")
    print(f"📊 Val:   {len(val_files)} scans ({len(val_subs)} subjects)")
    print(f"📊 Test:  {len(test_files)} scans ({len(test_subs)} subjects)")
    print(f"test subjects: {test_subs}")

    return train_files, val_files, test_files
    
def split_by_subject_n_folds(paired_files, subject_to_files, 
                         n_folds=5, fold_index=0, random_state=42):
    """
    按完整 Subject ID 划分为 n 折，返回指定 fold_index 的验证集，其余为训练集。
    """
    subject_ids = list(subject_to_files.keys())
    subject_ids.sort()  # 确保顺序一致

    # 计算每折的大小
    fold_size = len(subject_ids) // n_folds
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        if i == n_folds - 1:  # 最后一折包含剩余所有
            end_idx = len(subject_ids)
        else:
            end_idx = (i + 1) * fold_size
        folds.append(subject_ids[start_idx:end_idx])

    val_subs = set(folds[fold_index])
    train_subs = set(subject_ids) - val_subs

    def _filter_by_subjects(files_list, subject_set):
        subject_set = set(subject_set)
        return [f for f in files_list if extract_subject_id(os.path.basename(f[0])) in subject_set]

    train_files = _filter_by_subjects(paired_files, train_subs)
    val_files   = _filter_by_subjects(paired_files, val_subs)

    # 打印统计
    print(f"📊 Fold {fold_index+1}/{n_folds} - Train: {len(train_files)} scans, Val: {len(val_files)} scans")

    return train_files, val_files



