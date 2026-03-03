import os
from collections import Counter

from numpy import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import matplotlib
matplotlib.use('Agg') # 设置后端为Agg，这一行必须在import pyplot之前
import matplotlib.pyplot as plt
from umap import UMAP
import numpy as np
from sklearn import metrics, manifold
import scipy.io as scio
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve
)
from dataset.load_dataset import MRIPETDataset, get_paired_files_with_subjects, split_by_subject,load_data_from_optimized_csv
from resnet_concat_QKV2 import resnet18_3d
from configs import load_config
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os
os.environ['MPLBACKEND'] = 'Agg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
count = 1
c1 = 'AD'
c2 = 'MCI'
i = "resnet__QKV_concat_rel_kjrhjj"
ACC_SEN_THRESHOLD = 0.55
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: (N,2) logits, targets: (N,) in {0,1}
        log_probs = F.log_softmax(inputs, dim=1)          # (N,2)
        probs = log_probs.exp()                           # (N,2)

        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)       # (N,)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # per-sample alpha_t: class 1 gets alpha, class 0 gets (1-alpha)
        alpha_t = torch.where(targets == 1,
                              torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype),
                              torch.tensor(1.0 - self.alpha, device=inputs.device, dtype=inputs.dtype))

        loss = -alpha_t * ((1 - pt) ** self.gamma) * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
def save_checkpoint(model, optimizer, args, epoch, acc, sen, spe, auc,
                    c1, c2, i, count, output_dir, suffix=''):
    filename = f'i_{i}_count_{count}{suffix}.pth'  # 将 = 改为 _
    save_path = os.path.join(output_dir, filename)
    
    # 确保目录真正可写
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f'✅ Created directory: {output_dir}')
    
    print(f'\n{"=" * 70}')
    print(f'💾 Saving Best Model...')
    print(f'   Path: {save_path}')
    print(f'   Epoch: {epoch}')
    print(f'   Acc: {acc:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}, AUC: {auc:.4f}')
    print(f'{"=" * 70}\n')
    
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    # 使用临时文件避免写入失败导致损坏
    import tempfile
    temp_path = save_path + '.tmp'
    
    try:
        torch.save({
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'acc': acc,
            'sen': sen,
            'spe': spe,
            'auc': auc,
            'c1': c1,
            'c2': c2,
            'fold': i,
            'count': count,
            'args': args,
        }, temp_path)
        
        # 如果临时文件写入成功，替换原文件
        remove_status = False
        while not remove_status:
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                remove_status = True
            except Exception as e:
                print(f'⚠️  Warning: Failed to remove existing file: {e}. Retrying...')
                time.sleep(0.1)
        
        os.rename(temp_path, save_path)
        print(f'✅ Model saved successfully!')
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f'❌ Failed to save model: {e}')
        raise
    
    return save_path
def _train(epoch, train_loader, model, optimizer, criterion_cls, args):
    model.train()
    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.long().to(device)
        data= F.interpolate(data, [128, 128, 128], mode='trilinear')
        optimizer.zero_grad()
        output, features = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)
        loss = criterion_cls(output, target)
        losses += loss.item()
        loss.backward()
        optimizer.step()
    print('train [Epoch: {0:4d}], Loss: {1:.3f}, Acc: {2:.3f}, Correct {3} / Total {4}'.format(
        epoch, losses / len(train_loader), acc / total * 100, acc, total))
    loss = losses / len(train_loader)
    return loss

def _eval(epoch, eval_loader, model, criterion_cls, args):
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []  # 存储所有类别的概率
    losses = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(eval_loader):
            if args.cuda:
                data, target = data.to(device), target.long().to(device)
            data = F.interpolate(data, [128, 128, 128], mode='trilinear')
            output,features = model(data)
            probs = F.softmax(output, dim=-1)
            _, pred = probs.max(1)
            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_probs.append(probs.cpu())
            # 计算验证集loss
            loss = criterion_cls(output, target)
            losses += loss.item()
        # 拼接所有batch
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_scores = torch.cat(all_probs).numpy()
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
    acc = accuracy_score(all_targets, all_preds)
    sen = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    spe = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
    auc = roc_auc_score(all_targets, all_scores[:, 1])
    val_loss = losses / len(eval_loader)
    # 打印结果
    print(f'Eval: [Epoch: {epoch:4d}], Loss: {val_loss:.3f}')
    print(f'Acc: {acc:.4f}, Sen: {sen:.4f}, Spe: {spe:.4f}, AUC: {auc:.4f}')
    return acc,sen,spe,auc,val_loss

def _test(test_loader, model, args):
    """
    测试函数
    返回：acc, sen, spe, auc, fpr, tpr, all_features, all_targets, all_preds, all_probs
    """
    model.eval()
    all_preds = []  # 预测标签
    all_targets = []  # 真实标签
    all_probs = []  # 预测概率（用于ROC）
    all_features = []  # 特征（用于UMAP）
    
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.to(device), target.long().to(device)
            data = F.interpolate(data, [128, 128, 128], mode='trilinear')
            
            # 前向传播
            output, features = model(data)
            
            # 获取预测结果
            probs = F.softmax(output, dim=-1)  # 概率分布
            _, pred = probs.max(1)  # 预测类别
            
            # 收集结果（转换为numpy数组）
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())
            
    # 合并所有批次的结果
    all_preds = np.concatenate(all_preds)  # shape: (n_samples,)
    all_targets = np.concatenate(all_targets)  # shape: (n_samples,)
    all_probs = np.vstack(all_probs)  # shape: (n_samples, n_classes)
    all_features = np.vstack(all_features)  # shape: (n_samples, feature_dim)
    
    acc = accuracy_score(all_targets, all_preds)
    sen = recall_score(all_targets, all_preds, pos_label=1, zero_division=0)
    spe = recall_score(all_targets, all_preds, pos_label=0, zero_division=0)
    auc = roc_auc_score(all_targets, all_probs[:, 1])
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs[:, 1], pos_label=1)
    print(f'Accuracy: {acc:.4f}')
    print(f'Sensitivity: {sen:.4f}')
    print(f'Specificity: {spe:.4f}')
    print(f'AUC: {auc:.4f}')
    return acc, sen, spe, auc, fpr, tpr, all_features, all_targets, all_preds, all_probs
def main(args):
    global count, c1, c2
    '''load and process dataset'''
    '''split dataset and train'''
    # 统一输出目录
    output_dir = os.path.join(r'D:\zxyself\output', f'{c1}_vs_{c2}', 'resnet18_QKV_concat_rel_kjrhjj')
    os.makedirs(output_dir, exist_ok=True)
    print(f'📁 All outputs will be saved to: {output_dir}\n')
    
    for i in [6]:
        train_loss = []
        # paired_files, subject_to_files = get_paired_files_with_subjects("D:\zxyself\zxyself_traindata\AD_MCI2\MRI_Normalized",
        #                                                                 "D:\zxyself\zxyself_traindata\AD_MCI2\PET_Normalized",
        #                                                                 "D:\zxyself\zxyself_traindata\AD_MCI2\mri_2_pet_mapping.csv")
        # train_files, val_files, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8,
        #                                                       val_ratio=0.1)
        mri_dir = r"C:\Users\5090-13\Desktop\zxycode1\zxypaper2\zxyself_traindata\zxyself_traindata\AD_MCI2\MRI_Normalized"
        pet_dir = r"C:\Users\5090-13\Desktop\zxycode1\zxypaper2\zxyself_traindata\zxyself_traindata\AD_MCI2\PET_Normalized"
        
        # 🔥 关键修改：这里要指向你新生成的 optimized_dataset_split.csv
        # 使用目录中的具体文件 optimized_dataset_split.csv
        csv_path = r"C:\Users\5090-13\Desktop\zxycode1\zxypaper2\output\output\AD_vs_MCI\optimized_dataset_split4.csv"
        # 2. 调用新函数直接获取分好的列表 (不再使用 split_by_subject)
        train_files, val_files, test_files = load_data_from_optimized_csv(csv_path, mri_dir, pet_dir)
        # ============================================================
        # 🔥 新增功能 1：保存训练集、验证集、测试集的文件列表到 CSV
        # ============================================================
        print("📝 Saving dataset split info to CSV...")
        split_data_list = []
        
        # 辅助函数：提取信息并添加到列表
        def add_files_to_list(file_list, set_name):
            for item in file_list:
                # item 结构通常是 (mri_path, pet_path, label)
                # 使用 os.path.basename 只保留文件名，不保留绝对路径，看起来更整洁
                mri_name = os.path.basename(item[0])
                pet_name = os.path.basename(item[1])
                group = item[2]
                split_data_list.append({
                    'MRI': mri_name,
                    'PET': pet_name,
                    'Research Group': group,
                    'Set': set_name  # 这一列标记是 Train, Val 还是 Test
                })

        add_files_to_list(train_files, 'Train')
        add_files_to_list(val_files, 'Validation')
        add_files_to_list(test_files, 'Test')
        
        # 转换为 DataFrame 并保存
        df_splits = pd.DataFrame(split_data_list)
        split_csv_path = os.path.join(output_dir, f'fold_{i}_dataset_splits.csv')
        df_splits.to_csv(split_csv_path, index=False)
        print(f"✅ Dataset splits saved to: {split_csv_path}")
        # ============================================================
        
        # 创建 Dataset 实例
        train_dataset = MRIPETDataset(train_files)
        val_dataset = MRIPETDataset(val_files)
        test_dataset = MRIPETDataset(test_files)
        
        label_map = {'MCI': 0, 'AD': 1}
        # 检查标签分布
        train_labels = [label_map[label] for _, _, label in train_files]
        val_labels = [label_map[label] for _, _, label in val_files]
        test_labels = [label_map[label] for _, _, label in test_files]
        print(f'\n📊 Train label distribution: {Counter(train_labels)}')
        print(f'📊 Val label distribution: {Counter(val_labels)}')
        print(f'📊 Test label distribution: {Counter(test_labels)}\n')
        
        # 设置随机种子
        torch.manual_seed(1)
        # 创建 DataLoader 实例
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        model = resnet18_3d(num_classes=2, input_channels=1)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        start_epoch = 1
        criterion_cls = FocalLoss(0.69,2)
        if args.cuda:
            model = model.to(device)
            criterion_cls = criterion_cls.to(device)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        global_acc = 0.
        global_sen = 0.
        global_spe = 0.
        global_auc = 0.
        val_loss = []
        best_score = -float('inf')
        saved_checkpoints = []
        for epoch in range(start_epoch, args.epochs + 1):
            # 训练
            loss = _train(epoch, train_loader, model, optimizer,
                          criterion_cls,args)
            train_loss.append(loss)
            # 验证
            current_acc, current_sen, current_spe, current_auc, current_val_loss = _eval(
                epoch, val_loader, model, criterion_cls, args)
            val_loss.append(current_val_loss)
             # 🔥 计算综合分数
            current_score = (current_acc + current_sen + current_spe + current_auc) / 4
            if current_score > best_score:
                best_score = current_score
                global_acc = current_acc
                global_sen = current_sen
                global_spe = current_spe
                global_auc = current_auc

            meets_threshold = (
                current_acc >= ACC_SEN_THRESHOLD and
                current_sen >= ACC_SEN_THRESHOLD 
            )
            if meets_threshold:
                suffix = f'_epoch_{epoch}_acc_{current_acc:.3f}_sen_{current_sen:.3f}_spe_{current_spe:.3f}_auc_{current_auc:.3f}'
                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    args=args,
                    epoch=epoch,
                    acc=current_acc,
                    sen=current_sen,
                    spe=current_spe,
                    auc=current_auc,
                    c1=c1,
                    c2=c2,
                    i=i,
                    count=count,
                    output_dir=output_dir,
                    suffix=suffix
                )
                saved_checkpoints.append({
                    'epoch': epoch,
                    'path': ckpt_path,
                    'val_acc': current_acc,
                    'val_sen': current_sen,
                    'val_spe': current_spe,
                    'val_auc': current_auc
                })
                print(f'✅ Checkpoint saved (Acc {current_acc:.3f}, Sen {current_sen:.3f}) meets threshold {ACC_SEN_THRESHOLD}')
            # 学习率调整
            lr_scheduler.step()
            print(f'Current Learning Rate: {lr_scheduler.get_last_lr()}')
        # 训练结束，打印最佳结果
        print(f'Best Eval: Acc: {global_acc:.3f}, Sen: {global_sen:.4f}, '
              f'Spe: {global_spe:.4f}, AUC: {global_auc:.4f}')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        epochs_range = range(1, len(train_loss) + 1)
        plt.plot(epochs_range, train_loss, 'b-', label='Train Loss', linewidth=2)
        plt.plot(epochs_range, val_loss, 'r-', label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Training and Validation Loss ({c1} vs {c2})', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        loss_curve_path = os.path.join(output_dir, f'i_{i}_loss_curve.png')
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Loss curve saved to: {loss_curve_path}')
        
        if not saved_checkpoints:
            print(f'⚠️ No checkpoints met Acc and Sen >= {ACC_SEN_THRESHOLD}; skip test evaluation.')
        else:
            summary_rows = []
            for ckpt in saved_checkpoints:
                checkpoints = torch.load(ckpt['path'], weights_only=False)
                model.load_state_dict(checkpoints['model_state_dict'])
                acc, sen, spe, auc, fpr, tpr, test_features, test_targets, test_preds, test_probs = _test(
                    test_loader, model, args)

                result_file = os.path.join(output_dir, f'i_{i}_epoch_{ckpt["epoch"]}_test_results.txt')
                with open(result_file, 'w', encoding='utf-8') as f:
                    f.write(f'Test Results for {c1} vs {c2} (Fold {i})\n')
                    f.write(f'{"="*50}\n')
                    f.write(f'Val Acc:      {ckpt["val_acc"]:.4f}\n')
                    f.write(f'Val Sen:      {ckpt["val_sen"]:.4f}\n')
                    f.write(f'Val Spe:      {ckpt["val_spe"]:.4f}\n')
                    f.write(f'Val AUC:      {ckpt["val_auc"]:.4f}\n')
                    f.write(f'{"-"*50}\n')
                    f.write(f'Test Accuracy:    {acc:.4f}\n')
                    f.write(f'Test Sensitivity: {sen:.4f}\n')
                    f.write(f'Test Specificity: {spe:.4f}\n')
                    f.write(f'Test AUC:         {auc:.4f}\n')
                    f.write(f'{"="*50}\n')
                    f.write(f'Model Path: {ckpt["path"]}\n')
                    f.write(f'Checkpoint Epoch: {ckpt["epoch"]}\n')
                print(f'✅ Test results saved to: {result_file}')

                summary_rows.append({
                    'epoch': ckpt['epoch'],
                    'checkpoint': os.path.basename(ckpt['path']),
                    'val_acc': ckpt['val_acc'],
                    'val_sen': ckpt['val_sen'],
                    'val_spe': ckpt['val_spe'],
                    'val_auc': ckpt['val_auc'],
                    'test_acc': acc,
                    'test_sen': sen,
                    'test_spe': spe,
                    'test_auc': auc
                })

            summary_df = pd.DataFrame(summary_rows)
            summary_csv_path = os.path.join(output_dir, f'i_{i}_test_summary.csv')
            summary_df.to_csv(summary_csv_path, index=False)
            print(f'✅ Test summary saved to: {summary_csv_path}')
        
        
        # # ============================================================
        # # 🔥 新增功能 2：保存测试集详细预测结果到 CSV
        # # ============================================================
        # print("📝 Saving test predictions details to CSV...")
        
        # test_results_list = []
        # # test_files 的顺序和 test_loader 是一致的 (因为 shuffle=False)
        # # test_preds, test_probs 的顺序也是一致的
        
        # # 定义标签反向映射，方便看结果 (0->CN, 1->MCI)
        # # 注意：这里要确保和你 label_map 定义的一致
        # idx_to_label = {0: 'MCI', 1: 'AD'} 
        
        # for idx, file_info in enumerate(test_files):
        #     mri_name = os.path.basename(file_info[0])
        #     pet_name = os.path.basename(file_info[1])
        #     true_label_str = file_info[2] # 原始字符串标签
            
        #     # 获取模型预测
        #     pred_idx = test_preds[idx]
        #     pred_label_str = idx_to_label.get(pred_idx, str(pred_idx))
            
        #     # 获取预测概率 (假设第1列是 MCI 的概率)
        #     mci_prob = test_probs[idx, 1] 
            
        #     # 判断是否预测正确
        #     is_correct = (pred_idx == label_map[true_label_str])
            
        #     test_results_list.append({
        #         'MRI': mri_name,
        #         'PET': pet_name,
        #         'True Group': true_label_str,
        #         'Predicted Group': pred_label_str,
        #         'Prediction Correct': is_correct,
        #         'Probability (MCI)': f"{mci_prob:.4f}" # 保留4位小数
        #     })
            
        # df_test_preds = pd.DataFrame(test_results_list)
        # test_pred_path = os.path.join(output_dir, f'fold_{i}_test_prediction_details.csv')
        # df_test_preds.to_csv(test_pred_path, index=False)
        # print(f"✅ Test predictions saved to: {test_pred_path}")
        # # ============================================================
        
        # # 绘制平均ROC曲线
        # plt.plot(fpr, tpr,
        #          label="AUC={0}".format(auc),
        #          color='blue', linewidth=2)
        # # 绘制对角线
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc="lower right")
        # # 保存图形到文件
        # plt.savefig(os.path.join(output_dir, f'i_{i}_ROC_output.png'))
        # # 清除当前图形（可选，但推荐）
        # plt.close()
        # fpr_path = os.path.join(output_dir, f'i_{i}_fpr.mat')
        # tpr_path = os.path.join(output_dir, f'i_{i}_tpr.mat')
        # scio.savemat(fpr_path, {'Net_fpr': fpr})
        # scio.savemat(tpr_path, {'Net_tpr': tpr})
        # # UMAP降维和绘图
        # reducer = UMAP(
        #     n_components=2,
        #     n_neighbors=30,
        #     min_dist=0.3,
        #     metric='euclidean'
        # )
        # embedding = reducer.fit_transform(test_features)
        # # 创建图形
        # plt.figure(figsize=(10, 8))
        # # 分别绘制两类样本
        # colors = ['blue', 'red']
        # labels = ['MCI', 'AD']
        # for idx, (color, label) in enumerate(zip(colors, labels)):
        #     indices = np.where(test_targets == idx)[0]  # 获取索引而不是布尔掩码
        #     plt.scatter(embedding[indices, 0], embedding[indices, 1],
        #                 c=color,
        #                 label=label,
        #                 alpha=0.7,
        #                 s=100)

        # plt.xlabel('UMAP-1')
        # plt.ylabel('UMAP-2')
        # # plt.title('UMAP visualization of ROIC')
        # plt.legend()
        # # 保存图片
        # save_path = os.path.join(output_dir, f'i_{i}_umap_{c1}_vs_{c2}.png')
        # plt.savefig(save_path, dpi=600, bbox_inches='tight')
        # plt.close()

if __name__ == '__main__':
    args = load_config()
    main(args)
