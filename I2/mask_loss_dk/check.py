import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import os

def check_mask_size(mask_path, target_shape=(128, 128, 128)):
    print(f"{'='*20} 检查 Mask 尺寸 {'='*20}")
    print(f"文件路径: {mask_path}")
    
    if not os.path.exists(mask_path):
        print("错误: 文件不存在，请检查路径！")
        return

    # 1. 加载原始数据
    try:
        mask_obj = nib.load(mask_path)
        mask_data = mask_obj.get_fdata()
        print(f"\n[1] 原始文件 (Raw File)")
        print(f"    - 尺寸 (Shape): {mask_data.shape}")
        print(f"    - 数据类型: {mask_data.dtype}")
        print(f"    - 包含的标签值 (前10个): {np.unique(mask_data)[:10]}")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    # 2. 模拟 Dataset 中的裁剪 (Crop)
    # 对应 dataset.py 中的: mask = mask[3:179, 5:213, 0:176]
    # 注意：请确保你的 mask 尺寸足够大，否则这里会报错
    try:
        # 只有当原始尺寸符合预期（如 197x233x189）时才进行此裁剪
        # 或者你可以直接打印裁剪后的形状
        mask_cropped = mask_data[3:179, 5:213, 0:176]
        print(f"\n[2] 预处理裁剪后 (After Cropping [3:179, 5:213, 0:176])")
        print(f"    - 尺寸 (Shape): {mask_cropped.shape}")
    except Exception as e:
        print(f"\n[2] 裁剪失败 (可能原始尺寸太小): {e}")
        mask_cropped = mask_data # 失败则使用原图继续

    # 3. 模拟 Dataset 中的缩放 (Resize)
    # 对应 dataset.py 中的: F.interpolate(..., size=self.target_shape, mode='nearest')
    try:
        # 转换为 Tensor: (D, H, W) -> (1, 1, D, H, W) 用于 interpolate
        mask_tensor = torch.from_numpy(mask_cropped).float().unsqueeze(0).unsqueeze(0)
        
        # 使用最近邻插值
        mask_resized = F.interpolate(mask_tensor, size=target_shape, mode='nearest')
        
        # 去掉 batch 维度方便查看
        mask_final = mask_resized.squeeze().numpy()
        
        print(f"\n[3] 缩放后 (After Resizing to {target_shape})")
        print(f"    - 最终输入模型尺寸 (Shape): {mask_final.shape}")
        print(f"    - 验证标签是否损坏 (应为整数): {np.unique(mask_final)[:10]}")
        
    except Exception as e:
        print(f"\n[3] 缩放失败: {e}")

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 请在这里修改为你的 Mask 文件路径
    # ---------------------------------------------------------
    my_mask_path = r"C:\Users\5090-13\Desktop\zxycode\data1\mri_dk_mni\MRI_002_S_5178_I372850dk_atlas_in_mni.nii.gz"
    
    # 设置你想要训练的目标尺寸 (128 或 96)
    my_target_shape = (128, 128, 128) 

    check_mask_size(my_mask_path, my_target_shape)