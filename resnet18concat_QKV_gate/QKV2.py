import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalAttention3D(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.3, num_groups=32):
        super().__init__()
        
        # ... (基础定义省略) ...
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed = None

        # [Block 1] Cross Attention
        self.norm1_mri = nn.LayerNorm(embed_dim)
        self.norm1_pet = nn.LayerNorm(embed_dim)
        
        # 定义 Q, K, V ... (保持不变)
        self.mri_query = nn.Linear(embed_dim, embed_dim)
        self.pet_key = nn.Linear(embed_dim, embed_dim)
        self.pet_value = nn.Linear(embed_dim, embed_dim)
        
        self.pet_query = nn.Linear(embed_dim, embed_dim)
        self.mri_key = nn.Linear(embed_dim, embed_dim)
        self.mri_value = nn.Linear(embed_dim, embed_dim)
        
        self.mri_proj = nn.Linear(embed_dim, embed_dim)
        self.pet_proj = nn.Linear(embed_dim, embed_dim)

        # [Block 2] FFN
        self.norm2_mri = nn.LayerNorm(embed_dim)
        self.norm2_pet = nn.LayerNorm(embed_dim)
        # ... (MLP 定义保持不变) ...
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_mri = self._build_mlp(embed_dim, mlp_hidden_dim, dropout)
        self.mlp_pet = self._build_mlp(embed_dim, mlp_hidden_dim, dropout)

        # [Block 3] Spatial Fusion
        self.norm3_mri = nn.GroupNorm(num_groups, embed_dim)
        self.norm3_pet = nn.GroupNorm(num_groups, embed_dim)

        # 修正2: 最后一层 Conv 开启 bias=True，且使用 Dropout3d
        self.spatial_fusion_mri = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=True), 
            nn.Dropout3d(dropout) 
        )
        self.spatial_fusion_pet = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=True),
            nn.Dropout3d(dropout)
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def _build_mlp(self, dim, hidden_dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, feat_mri, feat_pet):
        B, C, D, H, W = feat_mri.shape
        N = D * H * W
        
        # Flatten & Pos Embed
        feat_mri_flat = feat_mri.flatten(2).transpose(1, 2)
        feat_pet_flat = feat_pet.flatten(2).transpose(1, 2)

        if self.pos_embed is None:
             self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim, device=feat_mri.device))
             nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        x_mri = feat_mri_flat + self.pos_embed
        x_pet = feat_pet_flat + self.pos_embed

        # ==========================================
        # Block 1: Cross Attention (Fixed Parallelism)
        # ==========================================
        
        # 修正1: 关键步骤！先缓存 Norm 后的状态
        # 这样双方都基于"这一轮开始前的状态"进行查询，避免信息泄露
        mri_norm_input = self.norm1_mri(x_mri)
        pet_norm_input = self.norm1_pet(x_pet)

        # --- Branch 1: MRI 查 PET ---
        q_mri = self.mri_query(mri_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_pet = self.pet_key(pet_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_pet = self.pet_value(pet_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mri = (q_mri @ k_pet.transpose(-2, -1)) * self.scale
        attn_mri = self.attn_dropout(attn_mri.softmax(dim=-1))
        x_attn_mri = (attn_mri @ v_pet).transpose(1, 2).reshape(B, N, C)
        
        # --- Branch 2: PET 查 MRI ---
        # 注意：这里用的 key/value 必须是 mri_norm_input (旧状态)，而不是更新后的 x_mri
        q_pet = self.pet_query(pet_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k_mri = self.mri_key(mri_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v_mri = self.mri_value(mri_norm_input).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn_pet = (q_pet @ k_mri.transpose(-2, -1)) * self.scale
        attn_pet = self.attn_dropout(attn_pet.softmax(dim=-1))
        x_attn_pet = (attn_pet @ v_mri).transpose(1, 2).reshape(B, N, C)

        # 并行更新残差
        x_mri = x_mri + self.dropout(self.mri_proj(x_attn_mri))
        x_pet = x_pet + self.dropout(self.pet_proj(x_attn_pet))

        # ==========================================
        # Block 2: FFN (Standard Serial)
        # ==========================================
        x_mri = x_mri + self.mlp_mri(self.norm2_mri(x_mri))
        x_pet = x_pet + self.mlp_pet(self.norm2_pet(x_pet))

        # ==========================================
        # Block 3: Spatial Fusion (Fixed Norm Order & Bias)
        # ==========================================
        
        # --- MRI Branch ---
        # 1. Reshape to 3D (Spatial Domain)
        x_mri_3d = x_mri.transpose(1, 2).reshape(B, C, D, H, W)
        # residual_mri = x_mri_3d # 残差基准
        
        # # 2. Norm (GroupNorm on 3D data)
        # x_mri_norm = self.norm3_mri(x_mri_3d)
        
        # # 3. Conv (With Bias & Dropout3d)
        # out_mri_spatial = self.spatial_fusion_mri(x_mri_norm)
        
        # # 4. Add
        # out_mri = residual_mri + out_mri_spatial
        
        # --- PET Branch ---
        x_pet_3d = x_pet.transpose(1, 2).reshape(B, C, D, H, W)
        # residual_pet = x_pet_3d
        
        # x_pet_norm = self.norm3_pet(x_pet_3d)
        # out_pet_spatial = self.spatial_fusion_pet(x_pet_norm)
        
        # out_pet = residual_pet + out_pet_spatial

        return x_mri_3d,x_pet_3d