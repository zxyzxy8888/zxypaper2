import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math
import torch.utils.checkpoint  # <--- 必须加上这一行
# -------------------------------------------------------------------------
# Part 1: 基础工具函数 (替代缺失的 .nn 和 .util)
# -------------------------------------------------------------------------


def unsqueeze_xdim(z, xdim):
    """
    让时间步参数 z (Batch,) 适配 3D 张量 (Batch, C, D, H, W)
    """
    # xdim 是 (Batch, C, D, H, W)，长度为 5
    # 我们需要将 z 扩展为 (Batch, 1, 1, 1, 1)
    dims_to_add = len(xdim) - 1
    shape = z.shape + (1,) * dims_to_add
    return z.reshape(shape)

def conv_nd(dims, *args, **kwargs):
    """
    根据维度创建卷积层。dims=3 时返回 Conv3d
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def avg_pool_nd(dims, *args, **kwargs):
    """
    根据维度创建平均池化层。
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def normalization(channels):
    """
    3D 推荐使用 GroupNorm，因为它对 BatchSize 不敏感 (3D 任务 BatchSize 通常很小)
    """
    return nn.GroupNorm(32, channels)

def zero_module(module):
    """
    将模型最后一层的权重初始化为0，这对 Diffusion 训练稳定性至关重要
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦时间步嵌入
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def checkpoint(func, inputs, params, flag):
    """
    简单的 Gradient Checkpointing 包装
    """
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs)
    else:
        return func(*inputs)

# -------------------------------------------------------------------------
# Part 2: 修正后的 3D UNet 组件 (Upsample/Downsample/Attention)
# -------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """Apply the module to `x` given `emb` timestep embeddings."""

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    修正版：支持 3D 各向同性缩放 (D, H, W 都变大)
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        # scale_factor=2 会自动处理 3D (D,H,W) -> (2D, 2H, 2W)
        x = F.interpolate(x, scale_factor=2, mode="trilinear")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    修正版：支持 3D 各向同性下采样
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 # 3D 卷积 stride=2 会同时减小 D, H, W
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

class LinearAttentionBlock3D(nn.Module):
    """
    轻量级线性注意力 (Linear Attention) - O(N) 复杂度
    适合 3D 高分辨率任务，显存占用极低，但拥有全局感受野。
    """
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        
        # 1x1 卷积生成 Q, K, V
        self.to_qkv = conv_nd(1, channels, channels * 3, 1)
        self.to_out = zero_module(conv_nd(1, channels, channels, 1))

    def _forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.shape
        x_flat = x.reshape(b, c, -1) # (B, C, N)
        n = x_flat.shape[-1]
        # 1. 生成 Q, K, V
        qkv = self.to_qkv(self.norm(x_flat))
        q, k, v = qkv.chunk(3, dim=1) # (B, C, N)
        # 2. 拆分 Heads
        # q, k, v -> (B, Heads, Dim, N)
        q = q.reshape(b, self.num_heads, c // self.num_heads, n)
        k = k.reshape(b, self.num_heads, c // self.num_heads, n)
        v = v.reshape(b, self.num_heads, c // self.num_heads, n)
        # 3. 线性注意力核心 trick: softmax(Q) * (softmax(K)^T * V)
        # 这里的 softmax 是沿着 spatial 维度做的
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        # 4. 先计算 K^T * V -> (B, Heads, Dim, Dim) -> 这是一个很小的全局上下文矩阵
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        # 5. 再计算 Q * Context -> (B, Heads, Dim, N)
        out = torch.einsum('bhdn,bhde->bhen', q, context)
        # 6. 还原形状
        out = out.reshape(b, c, n)
        out = self.to_out(out)
        
        return (x_flat + out).reshape(b, c, d, h, w)

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)
# -------------------------------------------------------------------------
# Part 3: 主 UNet 模型 (3D版)
# -------------------------------------------------------------------------

class UNetModel3D(nn.Module):
    def __init__(
        self,
        image_size,      # 例如 96 (假设是 96x96x96)
        in_channels,     # 输入通道数 (Concat后: 1个PET噪声 + 1个MRI = 2)
        model_channels,  # 基础通道数，例如 64 或 32 (3D建议小一点)
        out_channels,    # 输出通道数 (1个PET)
        num_res_blocks,  # 每个层级的ResBlock数量，例如 2
        attention_resolutions, # 哪一层使用 Attention (例如 [8,])
        dropout=0,
        channel_mult=(1, 2, 4, 8), # 通道倍增
        conv_resample=True,
        dims=3,          # <--- 强制设置为 3
        num_classes=None,
        use_checkpoint=True, # 3D 强烈建议开启 Gradient Checkpoint
        num_heads=4,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        # Downsample Path
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        LinearAttentionBlock3D(ch, num_heads=num_heads, use_checkpoint=use_checkpoint)
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            # Downsample logic
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # Middle Block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            # 如果显存不够，可以注释掉 Middle Block 的 Attention
            # AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            LinearAttentionBlock3D(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # Upsample Path
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        LinearAttentionBlock3D(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
                    )
                
                # Upsample Logic
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def forward(self, x, timesteps, cond=None):
        """
        x: Noisy PET (B, 1, D, H, W)
        cond: MRI Image (B, 1, D, H, W)
        timesteps: (B,)
        """
        # 如果有 Condition，在 Channel 维度拼接
        if cond is not None:
            x = torch.cat([x, cond], dim=1)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        h = x.type(torch.float32) # 保持精度
        
        # Down
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h, emb)
        
        # Up
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            
        return self.out(h)

# -------------------------------------------------------------------------
# Part 4: 对接你的 Network.py (I2SB 包装器)
# -------------------------------------------------------------------------

class Image3DNet(nn.Module):
    def __init__(self, use_fp16=False,use_attention=False):
        super().__init__()
        attn_res = (8,) if use_attention else ()
        
        # 配置你的 3D U-Net 参数
        self.diffusion_model = UNetModel3D(
            image_size=96,         # 输入尺寸，例如 (96, 96, 96)
            in_channels=2,         # 1 (Noisy PET) + 1 (MRI Condition)
            model_channels=64,     # 基础通道数，显存不够改 32
            out_channels=1,        # 输出 1 个 PET
            num_res_blocks=2,      # 每个层级 2 个 ResBlock
            attention_resolutions=attn_res, # 仅在最底层 (1/8) 使用 Attention
            channel_mult=(1, 2, 4, 8),  # 4层下采样
            dims=3,                # 核心：3D模式
            use_checkpoint=True,   # 核心：节省显存
            use_scale_shift_norm=True
        )
        
        if use_fp16:
            self.diffusion_model.half()

    def forward(self, x, steps, cond=None):
        """
        x: (B, 1, D, H, W) -> Noisy PET
        steps: (B,) -> Timesteps
        cond: (B, 1, D, H, W) -> MRI
        """
        # I2SB 的 diffusion.py 会传入 steps (int vector)
        # 我们不需要 noise_levels 查找表，直接传给 UNet 计算 embedding
        
        return self.diffusion_model(x, steps, cond=cond)

# -------------------------------------------------------------------------
# Part 5: 测试代码 (确保能跑)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模拟输入：Batch=1, Channel=1, Size=64x64x64
    x_noisy = torch.randn(1, 1, 96,96,96).to(device)
    x_mri   = torch.randn(1, 1, 96,96,96).to(device)
    timesteps = torch.tensor([500]).to(device) # 中间步数

    model = Image3DNet(use_attention=False).to(device)
    
    # 打印参数量
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    try:
        output = model(x_noisy, timesteps, cond=x_mri)
        print("Output shape:", output.shape) # 应该也是 (1, 1, 64, 64, 64)
        print("Success! 3D Model forward pass works.")
    except RuntimeError as e:
        print("Error (likely OOM or Dimension mismatch):", e)