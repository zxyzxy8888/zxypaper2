import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention3D(nn.Module):
    """
    3D医学影像自注意力模块 (Self-Attention Block)
    结构: Self-Attn -> FFN -> Spatial Fusion (Conv)
    """
    def __init__(self, embed_dim=512, num_heads=4, mlp_ratio=4.0, dropout=0.3, num_groups=32):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.pos_embed = None

        # ==========================================
        # Block 1: Self Attention
        # ==========================================
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Q, K, V 映射层 (注意：现在只需要一套)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # ==========================================
        # Block 2: FFN
        # ==========================================
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # ==========================================
        # Block 3: Spatial Fusion (3D Conv)
        # ==========================================
        # 使用 GroupNorm 适应 BS=16
        self.norm3 = nn.GroupNorm(num_groups, embed_dim)

        self.spatial_fusion = nn.Sequential(
            # Depthwise Conv3D
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, bias=False),
            nn.GroupNorm(num_groups, embed_dim),
            nn.GELU(),
            # Pointwise Conv3D (Bias=True)
            nn.Conv3d(embed_dim, embed_dim, kernel_size=1, bias=True), 
            # 使用 Dropout3d
            nn.Dropout3d(dropout) 
        )

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        输入: x [B, C, D, H, W]
        输出: x [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape
        N = D * H * W
        
        # 1. Flatten: [B, C, D, H, W] -> [B, N, C]
        x_flat = x.flatten(2).transpose(1, 2)

        # 2. Pos Embed (只初始化一次)
        if self.pos_embed is None:
             self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim, device=x.device))
             nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        x_flat = x_flat + self.pos_embed

        # ==========================================
        # Block 1: Self Attention (Pre-Norm)
        # ==========================================
        residual = x_flat
        x_norm = self.norm1(x_flat) # Pre-Norm

        # Q, K, V 都来自同一个 x_norm
        q = self.q_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention Calculation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_dropout(attn.softmax(dim=-1))
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        # Residual Add
        x_flat = residual + self.dropout(self.out_proj(x_attn))

        # ==========================================
        # Block 2: FFN
        # ==========================================
        residual = x_flat
        x_flat = residual + self.mlp(self.norm2(x_flat))

        # ==========================================
        # Block 3: Spatial Fusion
        # ==========================================
        # 1. Reshape back to 3D for Conv
        x_3d = x_flat.transpose(1, 2).reshape(B, C, D, H, W)
        # residual_3d = x_3d # 这里的残差是 3D 形式的
        
        # # 2. Pre-Norm (GroupNorm on Channels)
        # x_norm_3d = self.norm3(x_3d)
        
        # # 3. Conv + Dropout3d
        # x_spatial = self.spatial_fusion(x_norm_3d)
        
        # # 4. Residual Add
        # out = residual_3d + x_spatial

        return x_3d