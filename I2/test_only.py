import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 引入你的组件
from dataset import MRIPETDataset
from split_dataset import get_paired_files_with_subjects, split_by_subject
from performance_metric import mean_absolute_error, peak_signal_to_noise_ratio, structural_similarity_index

# --- 核心组件 ---
from model import Image3DNet  
from i2_diffusion import Diffusion  

# --- 工具函数 ---
def get_beta_schedule(beta_start, beta_end, num_diffusion_timesteps):
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

def normalize_to_neg_one_to_one(img):
    return img * 2.0 - 1.0

def unnormalize_to_zero_to_one(img):
    return (img + 1.0) / 2.0

def resize_volume(img, target_size=(96, 96, 96)):
    return F.interpolate(img, size=target_size, mode='trilinear', align_corners=False)

def main():
    parser = argparse.ArgumentParser(description='3D I2SB 纯测试脚本')
    # 数据集路径 (请保持和你训练时一致)
    parser.add_argument('--mri_dir', type=str, required=True)
    parser.add_argument('--pet_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    
    # 权重路径 (关键：指向你保存的 best_i2sb_model.pth)
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--output_dir', type=str, default="test_ema_results_adaml1", help='结果保存目录')
    
    # 采样参数 (建议保持和你训练时一致，例如 700)
    parser.add_argument('--timesteps', type=int, default=700, help='采样步数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_attention', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading model from {args.model_path}...")
    
    # 1. 数据准备 (必须和训练时的划分逻辑一致，否则会把训练集当测试集测)
    paired_files, subject_to_files = get_paired_files_with_subjects(args.mri_dir, args.pet_dir, args.csv_path)
    # 注意：这里的 random_seed 必须和 split_dataset.py 里的一样，默认通常是固定的
    train_files, val_files, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8, val_ratio=0.1)
    
    test_dataset = MRIPETDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    logger.info(f"Test set size: {len(test_files)}")

    # 2. 初始化 I2SB & 加载模型
    betas = get_beta_schedule(0.0001, 0.02, args.timesteps)
    diffusion = Diffusion(betas, device)
    
    model = Image3DNet(use_fp16=False, use_attention=args.use_attention).to(device)
    
    # 加载权重
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. 开始测试
    total_mae = 0
    total_psnr = 0
    total_ssim = 0
    
    logger.info(f"Starting Inference with {args.timesteps} steps...")
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            # 预处理
            mri_imgs_raw = batch[0].to(device)
            pet_imgs_raw = batch[1].to(device)
            
            mri_imgs = normalize_to_neg_one_to_one(resize_volume(mri_imgs_raw))
            pet_imgs = normalize_to_neg_one_to_one(resize_volume(pet_imgs_raw))
            
            # 定义预测函数
            def pred_x0_fn(xt, t):
                return model(xt, t, cond=mri_imgs)

            # 执行全量采样
            traj_x, traj_x0 = diffusion.ddpm_sampling(
                steps=list(range(args.timesteps)), # [0, 1, ... 699]
                pred_x0_fn=pred_x0_fn,
                x1=mri_imgs,
                verbose=False
            )
            
            # 获取结果并逆归一化
            generated_pet = traj_x0[:, 0]
            gen_0to1 = unnormalize_to_zero_to_one(generated_pet).clamp(0, 1)
            pet_0to1 = unnormalize_to_zero_to_one(pet_imgs)
            
            # 计算指标
            mae = mean_absolute_error(pet_0to1, gen_0to1)
            psnr = peak_signal_to_noise_ratio(pet_0to1, gen_0to1)
            ssim = structural_similarity_index(pet_0to1, gen_0to1)
            
            total_mae += mae.item()
            total_psnr += psnr.item()
            total_ssim += ssim.item()
            
            # (可选) 保存每一张生成的图以便后续写论文用
            # 这里我只保存前5张，你想全保存可以去掉 if i < 5
            if i < 5:
                import matplotlib.pyplot as plt
                slice_idx = 48 # 中间切片
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(unnormalize_to_zero_to_one(mri_imgs)[0,0,:,:,slice_idx].cpu(), cmap='gray'); axes[0].set_title('MRI')
                axes[1].imshow(gen_0to1[0,0,:,:,slice_idx].cpu(), cmap='gray'); axes[1].set_title(f'I2SB (MAE={mae:.3f})')
                axes[2].imshow(pet_0to1[0,0,:,:,slice_idx].cpu(), cmap='gray'); axes[2].set_title('Real PET')
                plt.savefig(os.path.join(args.output_dir, f"result_{i}.png"))
                plt.close()

    # 4. 汇总输出
    n = len(test_loader)
    logger.info(f"Final Test Results ({n} samples):")
    logger.info(f"MAE : {total_mae/n:.5f}")   # 看这个数能不能打败 U-Net
    logger.info(f"PSNR: {total_psnr/n:.4f}")
    logger.info(f"SSIM: {total_ssim/n:.4f}")  # 重点看这个

if __name__ == "__main__":
    main()