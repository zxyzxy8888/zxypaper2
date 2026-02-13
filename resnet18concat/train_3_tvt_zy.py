import os
from collections import Counter

from numpy import random
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
from dataset.load_dataset import MRIPETDataset, get_paired_files_with_subjects, split_by_subject
from configs import load_config
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import os
os.environ['MPLBACKEND'] = 'Agg'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
count = 1
c1 = 'AD'
c2 = 'MCI'
i = "resnet18"
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
                    c1, c2, i, count, suffix=''):
    save_dir = os.path.join(r'D:\zxyself\output', f'{c1}_vs_{c2}', args.output_subdir)
    os.makedirs(save_dir, exist_ok=True)
    filename = f'i={i}-{count}{suffix}.pth'
    save_path = os.path.join(save_dir, filename)
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
    }, save_path)
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


def _test(test_loader,model, args):
    """
    测试函数
    返回：acc, sen, spe, auc, fpr, tpr, all_features, all_labels
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
            output,features = model(data)
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
    # 4. ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs[:, 1], pos_label=1)
    # ==================== 打印结果 ====================
    print(f'Accuracy: {acc:.4f}')
    print(f'Sensitivity: {sen:.4f}')
    print(f'Specificity: {spe:.4f}')
    print(f'AUC: {auc:.4f}')
    # ==================== 返回结果 ====================
    return acc, sen, spe, auc, fpr, tpr, all_features, all_targets
def main(args):
    global count, c1, c2, m, path
    '''load and process dataset'''
    '''split dataset and train'''
    for i in [6]:
        train_loss = []
        paired_files, subject_to_files = get_paired_files_with_subjects("C:\\Users\\5090-13\\Desktop\\zxycode1\\zxypaper2\\zxyself_traindata\\zxyself_traindata\\AD_MCI2\\MRI_Normalized",
                                                                        "C:\\Users\\5090-13\\Desktop\\zxycode1\\zxypaper2\\zxyself_traindata\\zxyself_traindata\\AD_MCI2\\PET_Normalized",
                                                                        "C:\\Users\\5090-13\\Desktop\\zxycode1\\zxypaper2\\zxyself_traindata\\zxyself_traindata\\AD_MCI2\\mri_2_pet_mapping.csv")
        train_files, val_files, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8,
                                                              val_ratio=0.1)
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
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8,prefetch_factor=4,persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8,prefetch_factor=4,persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8,prefetch_factor=4,persistent_workers=True)
        from resnet_concat import resnet18_3d
        model = resnet18_3d(num_classes=2, input_channels=1)
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        start_epoch = 1
        criterion_cls = FocalLoss(alpha=0.75, gamma=2.0)
        if args.cuda:
            model = model.to(device)
            criterion_cls = criterion_cls.to(device)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        global_acc = 0.
        global_sen = 0.
        global_spe = 0.
        global_auc = 0.
        global_epoch = 0.
        val_loss = [] 
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
            best_score = (global_acc + global_sen + global_spe + global_auc) / 4
            # ✅ 保存规则：当前综合评分更高，则保存
            is_best = current_score > best_score
            if is_best:
                global_acc = current_acc
                global_sen = current_sen
                global_spe = current_spe
                global_auc = current_auc
                path = save_checkpoint(
                    model=model, optimizer=optimizer, args=args, epoch=epoch,
                    acc=current_acc, sen=current_sen, spe=current_spe, auc=current_auc,
                    c1=c1, c2=c2, i=i, count=count)
                print(f'✅ New best model saved! Score: {current_score:.4f} '
                      f'(Acc: {global_acc:.4f}, Sen: {global_sen:.4f}, '
                      f'Spe: {global_spe:.4f}, AUC: {global_auc:.4f})')
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
        output_dir = os.path.join(r'D:\zxyself\output', f'{c1}_vs_{c2}', args.output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        loss_curve_path = os.path.join(output_dir, f'i={i}_loss_curve.png')
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'✅ Loss curve saved to: {loss_curve_path}')
        
        # Test
        checkpoints = torch.load(path, weights_only=False)
        model.load_state_dict(checkpoints['model_state_dict'])
        acc, sen, spe, auc, fpr, tpr, test_features, test_targets= _test(
             test_loader, model, args)
        
        # 保存测试集结果到文件
        result_file = os.path.join(output_dir, f'i={i}_test_results.txt')
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f'Test Results for {c1} vs {c2} (Fold {i})\n')
            f.write(f'{"="*50}\n')
            f.write(f'Accuracy:    {acc:.4f}\n')
            f.write(f'Sensitivity: {sen:.4f}\n')
            f.write(f'Specificity: {spe:.4f}\n')
            f.write(f'AUC:         {auc:.4f}\n')
            f.write(f'{"="*50}\n')
            f.write(f'Model Path: {path}\n')
            f.write(f'Best Epoch: {checkpoints["epoch"]}\n')
        print(f'✅ Test results saved to: {result_file}')
        
        # 绘制平均ROC曲线
        plt.plot(fpr, tpr,
                 label="AUC={0}".format(auc),
                 color='blue', linewidth=2)
        # 绘制对角线
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        # 保存图形到文件
        plt.savefig(os.path.join(output_dir, f'i={i}ROC_output.png'))
        # 清除当前图形（可选，但推荐）
        plt.close()
        fpr_path = os.path.join(output_dir, f'i={i}_fpr.mat')
        tpr_path = os.path.join(output_dir, f'i={i}_tpr.mat')
        scio.savemat(fpr_path, {'Net_fpr': fpr})
        scio.savemat(tpr_path, {'Net_tpr': tpr})
        # UMAP降维和绘图
        reducer = UMAP(
            n_components=2,
            n_neighbors=50,
            min_dist=0.3,
            metric='euclidean'
        )
        embedding = reducer.fit_transform(test_features)
        # 创建图形
        plt.figure(figsize=(10, 8))
        # 分别绘制两类样本
        colors = ['blue', 'red']
        labels = ['MCI', 'AD']
        for idx, (color, label) in enumerate(zip(colors, labels)):
            indices = np.where(test_targets == idx)[0]  # 获取索引而不是布尔掩码
            plt.scatter(embedding[indices, 0], embedding[indices, 1],
                        c=color,
                        label=label,
                        alpha=0.7,
                        s=100)

        plt.xlabel('UMAP-1')
        plt.ylabel('UMAP-2')
        plt.title('UMAP visualization of Only_PET_Resnet18')
        plt.legend()
        # 保存图片
        save_path = os.path.join(output_dir,
                     f'umap_all_folds_{c1}_vs_{c2}.png')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    args = load_config()
    main(args)
