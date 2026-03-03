import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention3D(nn.Module):
    """
    3D医学影像的交叉模态注意力模块 (支持 Relative Positional Encoding)
    """
    def __init__(self, embed_dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.1, window_size=(4, 4, 4)):
        super().__init__()
        
        # ========== 基础参数 ==========
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size  # (D, H, W) 输入特征图的空间尺寸
        
        # ========== 相对位置编码 (Rel PE) ==========
        # 1. 定义相对位置偏置表
        # 范围是 [2*D-1, 2*H-1, 2*W-1]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # 2. 生成相对位置索引 (register_buffer 不会被视为模型参数更新)
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # [3, D, H, W]
        coords_flatten = torch.flatten(coords, 1)  # [3, N]
        
        # 计算相对坐标: relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 3]
        
        # 偏移坐标使其从0开始
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        # 计算乘数因子，将3D坐标映射到1D索引
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        
        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化偏置表
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

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
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim)
        )
        self.spatial_fusion_pet = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm3d(embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm3d(embed_dim)
        )
        
        # ========== Dropout ==========
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feat_mri, feat_pet):
        """
        Args:
            feat_mri: [B, C, D, H, W] 必须匹配初始化时的 window_size
            feat_pet: [B, C, D, H, W]
        """
        B, C, D, H, W = feat_mri.shape
        
        # 检查输入尺寸是否匹配 RelPE 的预设尺寸
        assert (D, H, W) == self.window_size, \
            f"Input size ({D},{H},{W}) must match window_size {self.window_size} for RelPE"

        # ========== 1. 展平为序列 ==========
        # 注意：不再需要加绝对位置编码 pos_embed
        feat_mri_flat = feat_mri.flatten(2).permute(0, 2, 1)  # [B, N, C]
        feat_pet_flat = feat_pet.flatten(2).permute(0, 2, 1)  # [B, N, C]
        
        N = feat_mri_flat.shape[1]
        
        # ========== 2. 准备相对位置偏置 ==========
        # 查表得到偏置: [N*N, num_heads] -> [N, N, num_heads] -> [num_heads, N, N]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # [N, N, nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, N, N]
        relative_position_bias = relative_position_bias.unsqueeze(0) # [1, nH, N, N]

        # ========== 3. 交叉注意力: MRI←PET ==========
        mri_q = self.mri_query(feat_mri_flat)  # [B, N, C]
        pet_k = self.pet_key(feat_pet_flat)    # [B, N, C]
        pet_v = self.pet_value(feat_pet_flat)  # [B, N, C]
        
        mri_q = mri_q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        pet_k = pet_k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        pet_v = pet_v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算 Attention Scores
        attn_mri = (mri_q @ pet_k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        
        # ✅ 注入相对位置偏置
        attn_mri = attn_mri + relative_position_bias
        
        attn_mri = F.softmax(attn_mri, dim=-1)
        attn_mri = self.attn_dropout(attn_mri)
        
        feat_mri_attn = (attn_mri @ pet_v).permute(0, 2, 1, 3).reshape(B, -1, C)
        feat_mri_attn = self.mri_proj(feat_mri_attn)
        feat_mri_attn = self.dropout(feat_mri_attn)
        
        feat_mri_cross = self.norm_mri_1(feat_mri_flat + feat_mri_attn)
        
        # ========== 4. 交叉注意力: PET←MRI ==========
        pet_q = self.pet_query(feat_pet_flat)
        mri_k = self.mri_key(feat_mri_flat)
        mri_v = self.mri_value(feat_mri_flat)
        
        pet_q = pet_q.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mri_k = mri_k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mri_v = mri_v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn_pet = (pet_q @ mri_k.transpose(-2, -1)) * self.scale
        
        # ✅ 注入相对位置偏置 (共享同一个偏置表，因为空间结构是一样的)
        attn_pet = attn_pet + relative_position_bias
        
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
        feat_mri_3d = feat_mri_ffn.permute(0, 2, 1).reshape(B, C, D, H, W)
        feat_pet_3d = feat_pet_ffn.permute(0, 2, 1).reshape(B, C, D, H, W)
        
        # # ========== 7. 空间融合卷积 ==========
        # feat_mri_spatial = self.spatial_fusion_mri(feat_mri_3d)
        # feat_pet_spatial = self.spatial_fusion_pet(feat_pet_3d)
        
        # 残差连接
        enhanced_mri = feat_mri + feat_mri_3d
        enhanced_pet = feat_pet + feat_pet_3d
        
        return enhanced_mri, enhanced_pet