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
from performance_metric import mean_absolute_error, peak_signal_to_noise_ratio, structural_similarity_index

# --- 核心组件 ---
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
    parser = argparse.ArgumentParser(description='I2SB Ensemble 测试 (详细分析版)')
    parser.add_argument('--mri_dir', type=str, required=True)
    parser.add_argument('--pet_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help='best_i2sb_model.pth 的路径')
    
    # === 关键设置 ===
    parser.add_argument('--ensemble_times', type=int, default=5, help='每个病人生成多少次取平均')
    
    parser.add_argument('--timesteps', type=int, default=700, help='采样步数')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_attention', action='store_true')
    parser.add_argument('--save_csv', type=str, default='ensemble_results_test_1_28.csv', help='保存详细结果的CSV路径')
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 1. 数据准备
    logger.info("正在加载测试集...")
    paired_files, subject_to_files = get_paired_files_with_subjects(args.mri_dir, args.pet_dir, args.csv_path)
    train_files, val_files, test_files = split_by_subject(paired_files, subject_to_files, train_ratio=0.8, val_ratio=0.1)
    
    
    
    test_dataset = MRIPETDataset(test_files)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    logger.info(f"Test Subset Size: {len(test_files)} ")

    # 2. 初始化模型
    betas = get_beta_schedule(0.0001, 0.02, args.timesteps)
    diffusion = Diffusion(betas, device)
    
    model = Image3DNet(use_fp16=False, use_attention=args.use_attention).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 3. 开始测试
    total_mae = 0; total_psnr = 0; total_ssim = 0
    all_results_data = []
    logger.info(f"Start Ensemble Inference: Each subject sampled {args.ensemble_times} times.")
    def get_val_steps(total_steps, val_steps):
        # 生成 [0, 20, 40, ..., 980] 这样的 50 个点
        indices = np.linspace(0, total_steps - 1, val_steps, dtype=int)
        return sorted(list(set(indices)))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # 获取文件名 (Dataset 返回的 batch[3] 是 mri_path 的 tuple)
            mri_path_tuple = batch[3]
            current_filename = os.path.basename(mri_path_tuple[0])
            
            mri_imgs = normalize_to_neg_one_to_one(resize_volume(batch[0].to(device)))
            pet_imgs = normalize_to_neg_one_to_one(resize_volume(batch[1].to(device))) # Real PET (GT)
            # 准备好 GT 的 0-1 版本用于计算指标
            pet_0to1 = unnormalize_to_zero_to_one(pet_imgs)
            test_inference_steps = 50 
            test_step_indices = get_val_steps(args.timesteps, test_inference_steps)
            
            def pred_x0_fn(xt, t):
                return model(xt, t, cond=mri_imgs)

            # === 生成 N 次 ===
            generated_list = []
            single_maes = [] # 存储这N次每次单独的MAE
            single_psnr=[]
            single_ssim=[]
            
            logger.info(f"--- Processing {current_filename} (Subject {i+1}) ---")

            for r in tqdm(range(args.ensemble_times), desc=f"Sampling {current_filename}", leave=False):
                _, traj_x0 = diffusion.ddpm_sampling(
                    steps=test_step_indices,
                    pred_x0_fn=pred_x0_fn,
                    x1=mri_imgs,
                    verbose=False
                )
                
                # 获取单次生成结果
                single_pred = traj_x0[:, 0]
                generated_list.append(single_pred) 
                
                # --- 计算单次生成的 MAE ---
                single_pred_0to1 = unnormalize_to_zero_to_one(single_pred).clamp(0, 1)
                single_mae = mean_absolute_error(pet_0to1, single_pred_0to1).item()
                single_psnr_val = peak_signal_to_noise_ratio(pet_0to1, single_pred_0to1).item()
                single_ssim_val = structural_similarity_index(pet_0to1, single_pred_0to1).item()
                single_maes.append(single_mae)
                single_psnr.append(single_psnr_val)
                single_ssim.append(single_ssim_val)

                # 打印单次结果
                logger.info(f"    Sample {r+1}: MAE = {single_mae:.5f}")

            # === 计算两种 MAE ===
            
            # 1. 单次 MAE 的平均值 (Average of Single MAEs)
            # 代表：如果不做 Ensemble，模型平均每次表现如何
            avg_single_mae = sum(single_maes) / len(single_maes)
            
            # 2. Ensemble 后的 MAE (MAE of Ensemble Average)
            # 代表：把图取平均合成一张后，这张合成图表现如何（通常这里会变好！）
            avg_pred_img = torch.stack(generated_list).mean(dim=0)
            gen_ensemble_0to1 = unnormalize_to_zero_to_one(avg_pred_img).clamp(0, 1)
            
            mae_ensemble = mean_absolute_error(pet_0to1, gen_ensemble_0to1)
            psnr_ensemble = peak_signal_to_noise_ratio(pet_0to1, gen_ensemble_0to1)
            ssim_ensemble = structural_similarity_index(pet_0to1, gen_ensemble_0to1)
            
            # 计算提升幅度
            improvement = avg_single_mae - mae_ensemble.item()
            subject_data = {
                "Subject_ID": i + 1,
                "Filename": current_filename,
                "Ensemble_MAE": mae_ensemble.item(),
                "Ensemble_PSNR": psnr_ensemble.item(),
                "Ensemble_SSIM": ssim_ensemble.item(),
                "Avg_Single_MAE": avg_single_mae,
                "Improvement": improvement
            }
            for run_idx, run_mae in enumerate(single_maes):
                subject_data[f"Run_{run_idx+1}_MAE"] = run_mae
                subject_data[f"Run_{run_idx+1}_PSNR"] = single_psnr[run_idx]
                subject_data[f"Run_{run_idx+1}_SSIM"] = single_ssim[run_idx]
            
            all_results_data.append(subject_data)
            # === 打印最终对比结果 ===
            logger.info(f"Subject {i+1} [{current_filename}] Results:")
            logger.info(f"  > Avg Single MAE : {avg_single_mae:.5f} (单次生成能力的均值)")
            logger.info(f"  > Ensemble MAE   : {mae_ensemble.item():.5f} (融合后的指标)")
            logger.info(f"  > Improvement    : {improvement:.5f} (Ensemble 带来的提升)")
            logger.info(f"  > PSNR: {psnr_ensemble.item():.4f} | SSIM: {ssim_ensemble.item():.4f}")
            logger.info("-" * 40)
            
            total_mae += mae_ensemble.item()
            total_psnr += psnr_ensemble.item()
            total_ssim += ssim_ensemble.item()

    # 4. 最终汇总
    real_n = len(all_results_data)
    n = len(test_loader)
    logger.info("=" * 40)
    logger.info(f"Final Test Set Average (Ensemble Metrics):")
    logger.info(f"MAE : {total_mae/n:.5f}")
    logger.info(f"PSNR: {total_psnr/n:.4f}")
    logger.info(f"SSIM: {total_ssim/n:.4f}")
    logger.info("=" * 40)
    df = pd.DataFrame(all_results_data)
    cols = ["Subject_ID", "Filename", "Ensemble_MAE", "Ensemble_PSNR", "Ensemble_SSIM", "Avg_Single_MAE", "Improvement"]
    run_cols = [c for c in df.columns if c.startswith("Run_")]
    df = df[cols + run_cols]
    
    df.to_csv(args.save_csv, index=False)
    logger.info(f"Detailed results saved to: {args.save_csv}")
if __name__ == "__main__":
    main()