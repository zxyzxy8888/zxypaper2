import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger
from tqdm import tqdm
import pandas as pd

# 引入你的组件
from dataset import MRIPETDataset
from split_dataset import get_paired_files_with_subjects, split_by_subject
# 确保 performance_metric.py 中已经包含我之前提供给你的 Masked 函数
from performance_metric import (
    masked_mean_absolute_error, 
    masked_peak_signal_to_noise_ratio,
    masked_structural_similarity_index,
    mean_absolute_error, 
    peak_signal_to_noise_ratio, 
    structural_similarity_index
)

from model_linear_att import Image3DNet  
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
    parser = argparse.ArgumentParser(description='I2SB Ensemble + Masked Metrics 测试')
    parser.add_argument('--mri_dir', type=str, required=True)
    parser.add_argument('--pet_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--ensemble_times', type=int, default=5, help='每个病人生成次数')
    parser.add_argument('--timesteps', type=int, default=700)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--save_csv', type=str, default='masked_ensemble_results.csv')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 数据准备
    paired_files, subject_to_files = get_paired_files_with_subjects(args.mri_dir, args.pet_dir, args.csv_path)
    _, _, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8, val_ratio=0.1)
    
    # 注意：如果使用了 Patch-based 训练，测试时建议保持尺寸一致
    test_dataset = MRIPETDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型
    betas = get_beta_schedule(0.0001, 0.02, args.timesteps)
    diffusion = Diffusion(betas, device)
    model = Image3DNet(use_fp16=False, use_attention=args.use_attention).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    all_results_data = []
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            mri_path_tuple = batch[3]
            current_filename = os.path.basename(mri_path_tuple[0])
            
            # 准备数据与生成脑掩码
            # MRI 预处理后背景为 0，以此提取 Mask
            mri_raw = batch[0].to(device)
            mri_0to1 = resize_volume(mri_raw)
            brain_mask = (mri_0to1 > 0).float() # [1, 1, 96, 96, 96]
            
            pet_raw = batch[1].to(device)
            pet_0to1 = resize_volume(pet_raw) # Ground Truth PET
            
            mri_input = normalize_to_neg_one_to_one(mri_0to1)
            test_step_indices = sorted(list(set(np.linspace(0, args.timesteps - 1, 50, dtype=int))))
            
            def pred_x0_fn(xt, t):
                return model(xt, t, cond=mri_input)

            # === 开始 Ensemble 生成 ===
            generated_list = []
            logger.info(f"Processing {current_filename} (Mask Volume: {brain_mask.sum().item():.0f} voxels)")

            for r in range(args.ensemble_times):
                _, traj_x0 = diffusion.ddpm_sampling(
                    steps=test_step_indices, pred_x0_fn=pred_x0_fn, x1=mri_input, verbose=False
                )
                generated_list.append(traj_x0[:, 0])

            # === 计算融合结果与 Masked 指标 ===
            # 先求平均生成最稳的图
            avg_pred_img = torch.stack(generated_list).mean(dim=0)
            gen_ensemble_0to1 = unnormalize_to_zero_to_one(avg_pred_img).clamp(0, 1)
            
            # 计算 Masked 指标 (脑内)
            m_mae = masked_mean_absolute_error(pet_0to1, gen_ensemble_0to1, brain_mask).item()
            m_psnr = masked_peak_signal_to_noise_ratio(pet_0to1, gen_ensemble_0to1, brain_mask).item()
            m_ssim = masked_structural_similarity_index(pet_0to1, gen_ensemble_0to1, brain_mask)
            
            # 计算 Raw 指标 (全图，用于对比证明背景干扰)
            raw_mae = mean_absolute_error(pet_0to1, gen_ensemble_0to1).item()
            
            subject_data = {
                "Filename": current_filename,
                "Masked_MAE": m_mae,
                "Masked_PSNR": m_psnr,
                "Masked_SSIM": m_ssim,
                "Raw_MAE": raw_mae
            }
            all_results_data.append(subject_data)
            
            logger.info(f"  > [Ensemble] Masked MAE: {m_mae:.5f} | Raw MAE: {raw_mae:.5f}")
            logger.info(f"  > [Ensemble] Masked PSNR: {m_psnr:.4f} | Masked SSIM: {m_ssim:.4f}")
            logger.info("-" * 40)

    # 4. 保存详细结果
    df = pd.DataFrame(all_results_data)
    df.to_csv(args.save_csv, index=False)
    logger.info(f"Final Avg Masked MAE: {df['Masked_MAE'].mean():.5f}")

if __name__ == "__main__":
    main()