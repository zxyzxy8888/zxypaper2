import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention3D(nn.Module):
    """
    3D医学影像的交叉模态注意力模块
    支持MRI-PET双向交叉注意力
    """
    def __init__(self, embed_dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.15):
        super().__init__()
        
        # ========== 基础参数 ==========
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # ========== 位置编码 ==========
        # 注意：实际的num_patches将在forward中动态计算
        # 初始化为None，第一次forward时会自动创建
        self.pos_embed = None
        self.pos_embed_initialized = False
        
        # ========== 交叉注意力层 ==========
        # MRI as Query, PET as Key/Value
        self.mri_query = nn.Linear(embed_dim, embed_dim)
        self.pet_key = nn.Linear(embed_dim, embed_dim)
        self.pet_value = nn.Linear(embed_dim, embed_dim)
        
        # PET as Query, MRI as Key/Value
        self.pet_query = nn.Linear(embed_dim, embed_dim)
        self.mri_key = nn.Linear(embed_dim, embed_dim)
        self.mri_value = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.mri_proj = nn.Linear(embed_dim, embed_dim)
        self.pet_proj = nn.Linear(embed_dim, embed_dim)
        
        # ========== Normalization ==========
        self.norm_mri_1 = nn.LayerNorm(embed_dim)
        self.norm_pet_1 = nn.LayerNorm(embed_dim)
        self.norm_mri_2 = nn.LayerNorm(embed_dim)
        self.norm_pet_2 = nn.LayerNorm(embed_dim)
        
        # ========== FFN (Feed-Forward Network) ==========
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_mri = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.mlp_pet = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # ========== 空间融合卷积 (Depthwise Separable) ==========
        self.spatial_fusion_mri = nn.Sequential(
            # Depthwise卷积：空间聚合
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, 
                      groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            # Pointwise卷积：通道混合
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim)
        )
        self.spatial_fusion_pet = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1,
                      groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim)
        )
        
        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # ========== 初始化 ==========
        self._init_weights()
    
    def _init_weights(self):
        """✅ 修复后的权重初始化"""
        # 位置编码用截断正态分布（在forward时动态创建，这里跳过）
        # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 遍历所有模块
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Linear层用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm的标准初始化
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.Conv3d):
                # ✅ Conv3d用Kaiming初始化
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm3d):
                # ✅ BatchNorm单独处理（权重是1维的！）
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, feat_mri, feat_pet):
        """
        Args:
            feat_mri: [B, C, D, H, W] = [B, 512, 4, 4, 4]
            feat_pet: [B, C, D, H, W] = [B, 512, 4, 4, 4]
        
        Returns:
            enhanced_mri: [B, C, D, H, W]
            enhanced_pet: [B, C, D, H, W]
        """
        B, C, D, H, W = feat_mri.shape
        
        # ========== 1. 展平为序列 ==========
        feat_mri_flat = feat_mri.flatten(2).permute(0, 2, 1)  # [B, N, C]
        feat_pet_flat = feat_pet.flatten(2).permute(0, 2, 1)  # [B, N, C]
        
        N = feat_mri_flat.shape[1]  # 动态计算patch数量
        
        # ========== 2. 添加位置编码 ==========
        # 第一次forward时动态初始化pos_embed
        if not self.pos_embed_initialized:
            self.pos_embed = nn.Parameter(
                torch.randn(1, N, self.embed_dim, device=feat_mri.device, dtype=feat_mri.dtype)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            self.pos_embed_initialized = True
        
        feat_mri_pos = feat_mri_flat + self.pos_embed  # [B, N, C]
        feat_pet_pos = feat_pet_flat + self.pos_embed
        
        # ========== 3. 交叉注意力: MRI←PET ==========
        # MRI作为Query，PET作为Key/Value
        mri_q = self.mri_query(feat_mri_pos)  # [B, N, C]
        pet_k = self.pet_key(feat_pet_pos)     # [B, N, C]
        pet_v = self.pet_value(feat_pet_pos)   # [B, N, C]
        
        # 重塑为多头
        mri_q = mri_q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, h, N, d]
        pet_k = pet_k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        pet_v = pet_v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力分数
        attn_mri = (mri_q @ pet_k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn_mri = F.softmax(attn_mri, dim=-1)
        attn_mri = self.attn_dropout(attn_mri)
        
        # 加权求和
        feat_mri_attn = (attn_mri @ pet_v).permute(0, 2, 1, 3).reshape(B, -1, C)  # [B, N, C]
        feat_mri_attn = self.mri_proj(feat_mri_attn)
        feat_mri_attn = self.dropout(feat_mri_attn)
        
        # 残差连接 + LayerNorm
        feat_mri_cross = self.norm_mri_1(feat_mri_flat + feat_mri_attn)
        
        # ========== 4. 交叉注意力: PET←MRI ==========
        pet_q = self.pet_query(feat_pet_pos)
        mri_k = self.mri_key(feat_mri_pos)
        mri_v = self.mri_value(feat_mri_pos)
        
        pet_q = pet_q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mri_k = mri_k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mri_v = mri_v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_pet = (pet_q @ mri_k.transpose(-2, -1)) * self.scale
        attn_pet = F.softmax(attn_pet, dim=-1)
        attn_pet = self.attn_dropout(attn_pet)
        
        feat_pet_attn = (attn_pet @ mri_v).permute(0, 2, 1, 3).reshape(B, -1, C)
        feat_pet_attn = self.pet_proj(feat_pet_attn)
        feat_pet_attn = self.dropout(feat_pet_attn)
        feat_pet_cross = self.norm_pet_1(feat_pet_flat + feat_pet_attn)
        # ========== 5. FFN ==========
        feat_mri_ffn = self.norm_mri_2(feat_mri_cross + self.mlp_mri(feat_mri_cross))
        feat_pet_ffn = self.norm_pet_2(feat_pet_cross + self.mlp_pet(feat_pet_cross))
        
        # ========== 6. Reshape回3D ==========
        feat_mri_3d = feat_mri_ffn.permute(0, 2, 1).reshape(B, C, D, H, W)  # [B, 512, 4, 4, 4]
        feat_pet_3d = feat_pet_ffn.permute(0, 2, 1).reshape(B, C, D, H, W)
        
        # # ========== 7. 空间融合卷积 ==========
        feat_mri_spatial = self.spatial_fusion_mri(feat_mri_3d)
        feat_pet_spatial = self.spatial_fusion_pet(feat_pet_3d)
        
        # 残差连接
        enhanced_mri = feat_mri + feat_mri_spatial  # 与空间融合后的输入残差
        enhanced_pet = feat_pet + feat_pet_spatial
        return enhanced_mri, enhanced_pet


# ========== 测试代码 ==========
if __name__ == "__main__":
    print("=" * 50)
    # 创建模型
    model = CrossModalAttention3D(
        embed_dim=512,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1
    )
    # 创建测试数据
    batch_size = 2
    feat_mri = torch.randn(batch_size, 512, 4, 4, 4)  # [B, C, D, H, W]
    feat_pet = torch.randn(batch_size, 512, 4, 4, 4)
    print("输入:")
    print(f"  MRI: {feat_mri.shape}")
    print(f"  PET: {feat_pet.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        enhanced_mri, enhanced_pet = model(feat_mri, feat_pet)
    
    print("\n输出:")
    print(f"  Enhanced MRI: {enhanced_mri.shape}")
    print(f"  Enhanced PET: {enhanced_pet.shape}")
    
    # 验证维度
    assert enhanced_mri.shape == feat_mri.shape, "MRI输出维度不匹配!"
    assert enhanced_pet.shape == feat_pet.shape, "PET输出维度不匹配!"
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n模型统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  参数量 (MB): {total_params * 4 / 1024 / 1024:.2f}")
    
    # 打印各层参数量
    print("\n各模块参数量:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {num_params:,}")
    print("\n" + "=" * 50)
    print("✅ 测试通过！模型初始化成功！")