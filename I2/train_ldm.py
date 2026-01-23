import warnings
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*deprecated.*")

import argparse
import os
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 引入你的组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import MRIPETDataset
from split_dataset import get_paired_files_with_subjects, split_by_subject
from performance_metric import mean_absolute_error, peak_signal_to_noise_ratio, structural_similarity_index

# --- 核心组件 ---
from model import Image3DNet  
from i2_diffusion import Diffusion  

def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

def normalize_to_neg_one_to_one(img):
    return img * 2.0 - 1.0

def unnormalize_to_zero_to_one(img):
    return (img + 1.0) / 2.0

def resize_volume(img, target_size=(96, 96, 96)):
    return F.interpolate(img, size=target_size, mode='trilinear', align_corners=False)

def main():
    parser = argparse.ArgumentParser(description='训练 3D I2SB 模型 (MRI -> PET)')
    parser.add_argument('--mri_dir', type=str, required=True, help='MRI目录')
    parser.add_argument('--pet_dir', type=str, required=True, help='PET目录')
    parser.add_argument('--csv_path', type=str, required=True, help='CSV路径')
    parser.add_argument('--output_dir', type=str, default=r"D:\zxyself\output", help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=50, help='训练轮数')
    
    # === 关键参数设置 ===
    parser.add_argument('--batch_size', type=int, default=2, help='物理显存允许的最大Batch Size')
    parser.add_argument('--grad_accum_steps', type=int, default=8, help='梯度累积步数 (等效BS = batch_size * grad_accum_steps)')
    
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--val_interval', type=int, default=1, help='验证间隔')
    parser.add_argument('--timesteps', type=int, default=500, help='扩散步数')
    parser.add_argument('--use_attention', action='store_true', help='是否开启Attention')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    
    args = parser.parse_args()
    # 兼容性别名：有些代码或用户可能使用 args.epochs
    args.epochs = args.num_epochs
    
    # 计算等效 Batch Size 以便日志记录
    effective_bs = args.batch_size * args.grad_accum_steps
    
    # 基础设置
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output = os.path.join(args.output_dir, 'I2SB_3D_output_attention')
    log_dir = os.path.join(output, 'logs')
    os.makedirs(output, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger.add(os.path.join(log_dir, f'{current_time}_i2sb.log'))
    logger.info(f"Start Training. Physical BS={args.batch_size}, Accum Steps={args.grad_accum_steps}, Effective BS={effective_bs}")
    
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"tensorboard_{current_time}"))
    
    # 1. 数据准备
    paired_files, subject_to_files = get_paired_files_with_subjects(args.mri_dir, args.pet_dir, args.csv_path)
    train_files, val_files, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8, val_ratio=0.1)
    
    train_dataset = MRIPETDataset(train_files) 
    val_dataset = MRIPETDataset(val_files)
    test_dataset = MRIPETDataset(test_files)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 2. 初始化 I2SB
    betas = get_beta_schedule(0.0001, 0.02, args.timesteps)
    diffusion = Diffusion(betas, device)
    
    # 3. 定义模型
    logger.info("Initializing 3D I2SB Model...")
    model = Image3DNet(use_fp16=False, use_attention=args.use_attention).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay)
    # 学习率调度器：余弦退火，最小 lr 为 1e-6，T_max 使用总轮数（args.epochs）
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # 4. 训练循环
    best_val_mae = float('inf')
    def get_val_steps(total_steps, val_steps):
        # 生成 [0, 20, 40, ..., 980] 这样的 50 个点
        indices = np.linspace(0, total_steps - 1, val_steps, dtype=int)
        return sorted(list(set(indices)))

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        
        # 这里的 step 是 enumerate 的索引
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.num_epochs}")
        
        # 每一轮开始前清空梯度 (防止上一轮残留)
        optimizer.zero_grad(set_to_none=True)
        
        for step, batch in progress_bar:
            # 1. 数据加载与预处理
            mri_imgs = normalize_to_neg_one_to_one(resize_volume(batch[0].to(device)))
            pet_imgs = normalize_to_neg_one_to_one(resize_volume(batch[1].to(device)))
            
            # 2. I2SB 前向传播
            t = torch.randint(0, args.timesteps, (mri_imgs.shape[0],), device=device).long()
            xt = diffusion.q_sample(t, x0=pet_imgs, x1=mri_imgs)
            pred_pet = model(xt, t, cond=mri_imgs)
            
            # 3. 计算 Loss
            # loss = F.mse_loss(pred_pet, pet_imgs)
            loss = F.l1_loss(pred_pet, pet_imgs)
            
            # === 修改核心：Loss 除以累积步数 ===
            # 这样多次累积后的梯度总和，才等效于一次大 Batch 的平均梯度
            loss = loss / args.grad_accum_steps
            loss.backward()
            
            # 4. 记录 Loss (为了显示好看，乘回去)
            epoch_loss += loss.item() * args.grad_accum_steps
            progress_bar.set_postfix({"loss": f"{loss.item() * args.grad_accum_steps:.4f}"})
            
            # === 修改核心：梯度更新 ===
            # 只有当 step 累积够了次数，或者到达由于数据集末尾时，才更新参数
            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) # 清空梯度，准备下一轮累积
                lr_scheduler.step()    
        writer.add_scalar("train/loss", epoch_loss / (step + 1), epoch)

        # 5. 验证循环 (保持不变)
        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_mae_sum = 0
            val_inference_steps = 50 
            val_step_indices = get_val_steps(args.timesteps, val_inference_steps)
            vis_generated = None; vis_mri = None; vis_real = None

            with torch.no_grad():
                for i, val_batch in enumerate(tqdm(val_loader, desc="Validating")):
                    mri_imgs = normalize_to_neg_one_to_one(resize_volume(val_batch[0].to(device)))
                    pet_imgs = normalize_to_neg_one_to_one(resize_volume(val_batch[1].to(device)))
                    
                    def pred_x0_fn(xt, t):
                        return model(xt, t, cond=mri_imgs)

                    traj_x, traj_x0 = diffusion.ddpm_sampling(
                        steps=val_step_indices,
                        pred_x0_fn=pred_x0_fn,
                        x1=mri_imgs, 
                        verbose=False
                    )
                    
                    generated_pet = traj_x0[:, 0]
                    gen_0to1 = unnormalize_to_zero_to_one(generated_pet).clamp(0, 1)
                    pet_0to1 = unnormalize_to_zero_to_one(pet_imgs)
                    
                    mae = mean_absolute_error(pet_0to1, gen_0to1)
                    val_mae_sum += mae.item()
                    
                    if i == 0:
                        vis_generated = gen_0to1; vis_mri = unnormalize_to_zero_to_one(mri_imgs); vis_real = pet_0to1
            
            avg_val_mae = val_mae_sum / len(val_loader)
            logger.info(f"Epoch {epoch} Val MAE: {avg_val_mae:.4f}")
            writer.add_scalar("val/MAE", avg_val_mae, epoch)
            
            if vis_generated is not None:
                mid = vis_generated.shape[4] // 2
                combined = torch.cat([vis_mri[0,:,:,:,mid], vis_generated[0,:,:,:,mid], vis_real[0,:,:,:,mid]], dim=2)
                writer.add_image("Val/MRI_Gen_Real", combined, epoch, dataformats='CHW')

            if avg_val_mae < best_val_mae:
                best_val_mae = avg_val_mae
                torch.save(model.state_dict(), os.path.join(output, "best_i2sb_model.pth"))
                logger.info(f"New Best Model Saved! MAE: {best_val_mae:.4f}")

    # 6. 测试集评估 (保持不变)
    logger.info("Starting Final Evaluation...")
    # 记得重新初始化模型并加载权重
    model = Image3DNet(use_fp16=False, use_attention=args.use_attention).to(device)
    model.load_state_dict(torch.load(os.path.join(output, "best_i2sb_model.pth")))
    model.eval()
    
    total_mae = 0; total_psnr = 0; total_ssim = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Test Set Eval"):
            mri_imgs = normalize_to_neg_one_to_one(resize_volume(batch[0].to(device)))
            pet_imgs = normalize_to_neg_one_to_one(resize_volume(batch[1].to(device)))
            
            def pred_x0_fn(xt, t):
                return model(xt, t, cond=mri_imgs)

            traj_x, traj_x0 = diffusion.ddpm_sampling(
                steps=list(range(args.timesteps)),
                pred_x0_fn=pred_x0_fn,
                x1=mri_imgs,
                verbose=False
            )
            generated_pet = traj_x0[:, 0]
            gen_0to1 = unnormalize_to_zero_to_one(generated_pet).clamp(0, 1)
            pet_0to1 = unnormalize_to_zero_to_one(pet_imgs)
            
            mae = mean_absolute_error(pet_0to1, gen_0to1)
            psnr = peak_signal_to_noise_ratio(pet_0to1, gen_0to1)
            ssim = structural_similarity_index(pet_0to1, gen_0to1) 
            
            total_mae += mae.item(); total_psnr += psnr.item(); total_ssim += ssim.item()
            mri_imgs = unnormalize_to_zero_to_one(mri_imgs)
            
            if i < 3:
                mid = gen_0to1.shape[4] // 2
                combined = torch.cat([mri_imgs[0,:,:,:,mid], gen_0to1[0,:,:,:,mid], pet_0to1[0,:,:,:,mid]], dim=2)
                writer.add_image(f"Test/Sample_{i}", combined , 0, dataformats='CHW')

    n_test = len(test_loader)
    logger.info(f"Test Results - MAE: {total_mae/n_test:.4f}")
    logger.info(f"Test Results - PSNR: {total_psnr/n_test:.4f}")
    logger.info(f"Test Results - SSIM: {total_ssim/n_test:.4f}")
    
    writer.close()

if __name__ == '__main__':
    main()