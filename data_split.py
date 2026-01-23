import pandas as pd
import os
import re
import random
import math

# ================= 配置区域 =================
# 第一次运行的文件
run1_split_csv = r'D:\zxyself\output\AD_vs_MCI\resnet18_QKV_gate_70143\fold_6_dataset_splits.csv'
run1_pred_csv = r'D:\zxyself\output\AD_vs_MCI\resnet18_QKV_gate_70143\fold_6_test_prediction_details.csv'

# 第二次运行的文件 (假设你换了随机种子跑了第二次，产生了fold_X...)
run2_split_csv = r'D:\zxyself\output\AD_vs_MCI\resnet18_QKV_gate_701\fold_6_dataset_splits.csv'
run2_pred_csv = r'D:\zxyself\output\AD_vs_MCI\resnet18_QKV_gate_701\fold_6_test_prediction_details.csv'

# 输出的新划分文件路径
output_new_split_csv = r'D:\zxyself\output\AD_vs_MCI\optimized_dataset_split2.csv'

# 设定比例
test_ratio_min = 0.1  # 测试集至少占 10%
val_ratio = 0.1       # 验证集占 10%
train_ratio = 0.8     # 训练集占 80%
# ===========================================

def get_subject_id(filename):
    """从文件名提取 Subject ID (例如 MRI_002_S_0729_I186073.nii.gz -> 002_S_0729)"""
    parts = filename.split('_')
    # 假设标准格式为 MRI_XXX_S_XXXX_...
    if len(parts) >= 4:
        return f"{parts[1]}_{parts[2]}_{parts[3]}"
    return "Unknown"

def load_data_and_extract_subjects(split_csv, pred_csv):
    """读取 CSV 并解析被试者状态"""
    df_split = pd.read_csv(split_csv)
    df_pred = pd.read_csv(pred_csv)
    
    # 1. 获取所有数据的完整信息（作为总数据库）
    # key: subject_id, value: list of file records (dict)
    subject_database = {} 
    
    # 记录该次运行中，每个 subject 属于哪个集 (Train/Val/Test)
    subj_set_map = {} 
    
    for _, row in df_split.iterrows():
        subj_id = get_subject_id(row['MRI'])
        if subj_id not in subject_database:
            subject_database[subj_id] = []
        
        # 存入文件信息，方便最后生成新 CSV
        subject_database[subj_id].append({
            'MRI': row['MRI'],
            'PET': row['PET'],
            'Research Group': row['Research Group']
        })
        subj_set_map[subj_id] = row['Set']

    # 2. 获取测试集中预测正确的 Subject
    # 注意：一个 subject 可能有多个图像，只要有一个图像预测错，我们通常认为该 subject "不稳定"
    # 或者策略：只有所有图像都预测对，才算该 subject 预测对。这里采用严格策略。
    
    # 先初始化所有在 pred_csv 里的 subject 为 True
    correct_subjects = set()
    failed_subjects = set()
    
    for _, row in df_pred.iterrows():
        subj_id = get_subject_id(row['MRI'])
        is_correct = row['Prediction Correct'] # bool or string "True"/"False"
        
        # 处理可能的类型差异
        if isinstance(is_correct, str):
            is_correct = (is_correct.lower() == 'true')
            
        if not is_correct:
            failed_subjects.add(subj_id)
        else:
            correct_subjects.add(subj_id)
            
    # 真正的 Correct 是：在 correct 集合中，且不在 failed 集合中 (全对)
    final_correct_subjects = correct_subjects - failed_subjects
    
    return subject_database, subj_set_map, final_correct_subjects

def main():
    print("🚀 开始重新划分数据集...")
    
    # 1. 加载两次运行的数据
    db1, set_map1, correct1 = load_data_and_extract_subjects(run1_split_csv, run1_pred_csv)
    db2, set_map2, correct2 = load_data_and_extract_subjects(run2_split_csv, run2_pred_csv)
    
    # 合并两个数据库的文件信息 (确保信息最全，以 run1 为主，补充 run2 独有的)
    # 理论上两个 split 文件的总被试者应该是一样的
    full_database = db1.copy()
    all_subjects = list(full_database.keys())
    total_subjects_count = len(all_subjects)
    
    print(f"📊 总被试者数量: {total_subjects_count}")
    
    # ================= 2. 构建新测试集 (New Test) =================
    target_test_count = math.ceil(total_subjects_count * test_ratio_min)
    print(f"🎯 目标测试集人数: {target_test_count} (>= {test_ratio_min*100}%)")
    
    # 策略 A: 两次都在 Test 集，且都预测正确 (Intersection)
    # 条件：Subject 在 Run1 Test 且 Correct, 且 在 Run2 Test 且 Correct
    tier1_candidates = []
    # 策略 B: 只要在某一次是 Test 且预测正确 (Union)
    tier2_candidates = []
    
    for subj in all_subjects:
        in_test1 = (set_map1.get(subj) == 'Test')
        in_test2 = (set_map2.get(subj) == 'Test')
        is_corr1 = (subj in correct1)
        is_corr2 = (subj in correct2)
        
        # Tier 1: 两次都是 Test 且都对
        if (in_test1 and is_corr1) and (in_test2 and is_corr2):
            tier1_candidates.append(subj)
        
        # Tier 2: 只要有一次是 (Test + Correct) 且没被归入 Tier 1
        elif (in_test1 and is_corr1) or (in_test2 and is_corr2):
            tier2_candidates.append(subj)
            
    print(f"   - Tier 1 (双稳健): {len(tier1_candidates)} 人")
    print(f"   - Tier 2 (单稳健): {len(tier2_candidates)} 人")
    
    new_test_subjects = []
    
    # 优先取 Tier 1
    new_test_subjects.extend(tier1_candidates)
    
    # 如果不够，从 Tier 2 补
    if len(new_test_subjects) < target_test_count:
        needed = target_test_count - len(new_test_subjects)
        # 随机从 Tier 2 选，或者按顺序选
        random.shuffle(tier2_candidates)
        take_more = tier2_candidates[:needed]
        new_test_subjects.extend(take_more)
        print(f"   - 从 Tier 2 补充了: {len(take_more)} 人")
    
    # 如果还不够 (极其罕见，说明模型太差了)，那就只能随机从剩下的人里补了
    # 这里假设模型不算太差，或者你接受测试集略小于 10% 如果真的分不对
    # 为了代码健壮性，我们可以从剩下的集合里随机补
    if len(new_test_subjects) < target_test_count:
        remaining_pool = list(set(all_subjects) - set(new_test_subjects))
        needed = target_test_count - len(new_test_subjects)
        take_random = random.sample(remaining_pool, needed)
        new_test_subjects.extend(take_random)
        print(f"⚠️ 警告: 稳健样本不足，随机补充了 {len(take_random)} 人")

    print(f"✅ 最终测试集人数: {len(new_test_subjects)}")

    # ================= 3. 构建新训练/验证集 (Train/Val) =================
    # 剩余的被试者
    remaining_subjects = list(set(all_subjects) - set(new_test_subjects))
    
    target_val_count = math.ceil(total_subjects_count * val_ratio)
    
    # 识别优先级
    # Train Priority: 两次都是 Train
    # Val Priority: 两次都是 Val
    train_priority = []
    val_priority = []
    others = []
    
    for subj in remaining_subjects:
        s1 = set_map1.get(subj)
        s2 = set_map2.get(subj)
        
        if s1 == 'Train' and s2 == 'Train':
            train_priority.append(subj)
        elif s1 == 'Validation' and s2 == 'Validation':
            val_priority.append(subj)
        else:
            others.append(subj)
            
    print(f"   - Train 优先池 (双Train): {len(train_priority)} 人")
    print(f"   - Val 优先池 (双Val): {len(val_priority)} 人")
    print(f"   - 待分配池 (混合/其他): {len(others)} 人")
    
    new_val_subjects = []
    new_train_subjects = []
    
    # --- 填充验证集 ---
    # 1. 先把 Val 优先的放进去
    new_val_subjects.extend(val_priority)
    
    # 2. 如果多了，剪裁（把多余的扔回 others，给训练集）- 但通常不会多太多，除非你验证集比例设很大
    if len(new_val_subjects) > target_val_count:
        # 截取前 target 个，剩下的扔给 Train (因为他们本来就是"两次验证"，给Train也没问题，防止数据泄露给Test就行)
        extras = new_val_subjects[target_val_count:]
        new_val_subjects = new_val_subjects[:target_val_count]
        new_train_subjects.extend(extras)
    
    # 3. 如果少了，从 others 里随机补
    elif len(new_val_subjects) < target_val_count:
        needed = target_val_count - len(new_val_subjects)
        if len(others) >= needed:
            random.shuffle(others)
            take = others[:needed]
            new_val_subjects.extend(take)
            others = others[needed:] # 移除已选的
        else:
            # others 都不够用了，这几乎不可能发生
            pass

    print(f"✅ 最终验证集人数: {len(new_val_subjects)}")
    
    # --- 填充训练集 ---
    # 1. Train 优先的肯定进 Train
    new_train_subjects.extend(train_priority)
    # 2. 剩下的 others 进 Train
    new_train_subjects.extend(others)
    
    print(f"✅ 最终训练集人数: {len(new_train_subjects)}")
    
    # ================= 4. 生成文件列表 =================
    final_rows = []
    
    # 辅助函数：将被试者对应的文件添加到列表
    def add_subj_files(sub_list, set_name):
        for subj in sub_list:
            files = full_database[subj]
            for f in files:
                final_rows.append({
                    'MRI': f['MRI'],
                    'PET': f['PET'],
                    'Research Group': f['Research Group'],
                    'Set': set_name
                })

    add_subj_files(new_train_subjects, 'Train')
    add_subj_files(new_val_subjects, 'Validation')
    add_subj_files(new_test_subjects, 'Test')
    
    # ================= 5. 保存 =================
    df_result = pd.DataFrame(final_rows)
    # 简单的打乱一下行顺序，看着自然一点（不影响训练，因为DataLoader会shuffle）
    df_result = df_result.sample(frac=1).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(output_new_split_csv), exist_ok=True)
    df_result.to_csv(output_new_split_csv, index=False)
    print(f"\n💾 新的数据集划分已保存至:\n{output_new_split_csv}")
    
    # 验证一下
    print("\n🔍 最终统计 (Images):")
    print(df_result['Set'].value_counts())

if __name__ == '__main__':
    # 确保随机性可复现（可选）
    random.seed(42)
    main()