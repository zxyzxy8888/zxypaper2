import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import math
import torch.utils.checkpoint

# -------------------------------------------------------------------------
# Part 1: 基础工具组件 (AttentionGate3D, Conv, Norm 等)
# -------------------------------------------------------------------------

class AttentionGate3D(nn.Module):
    """
    3D Attention Gate: 利用解码器信号 g 过滤编码器跳跃连接特征 x
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        # W_g: 处理解码器信号 (Gating Signal)
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        # W_x: 处理编码器特征 (Skip Connection)
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(8, F_int)
        )
        # psi: 生成注意力系数 (0~1)
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 空间对齐：如果 g 的尺寸比 x 小 (通常发生在下采样对齐时)，上采样 g
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        # 返回加权后的特征：背景噪声会被抑制
        return x * psi

def conv_nd(dims, *args, **kwargs):
    if dims == 3: return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"Unsupported dims: {dims}")

def normalization(channels):
    return nn.GroupNorm(32, channels)

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# -------------------------------------------------------------------------
# Part 2: 3D UNet 核心组件 (ResBlock, Upsample, Downsample)
# -------------------------------------------------------------------------

class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb): pass

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock): x = layer(x, emb)
            else: x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv: self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        if self.use_conv: x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=3, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv: self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)
        else: self.op = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, dims=3, use_checkpoint=False, use_scale_shift_norm=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(), conv_nd(dims, channels, self.out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels))
        self.out_layers = nn.Sequential(normalization(self.out_channels), nn.SiLU(), nn.Dropout(p=dropout), zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)))

        if self.out_channels == channels: self.skip_connection = nn.Identity()
        else: self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.use_checkpoint: return torch.utils.checkpoint.checkpoint(self._forward, x, emb)
        else: return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape): emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(3, channels, channels * 3, 1)
        self.proj_out = zero_module(conv_nd(3, channels, channels, 1))

    def forward(self, x):
        if self.use_checkpoint: return torch.utils.checkpoint.checkpoint(self._forward, x)
        else: return self._forward(x)

    def _forward(self, x):
        b, c, d, h, w = x.shape
        qkv = self.qkv(self.norm(x)).reshape(b, 3, self.num_heads, c // self.num_heads, -1)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = 1 / math.sqrt(math.sqrt(v.shape[2]))
        attn = torch.einsum("bhdn,bhdm->bhnm", q * scale, k * scale)
        attn = torch.softmax(attn.float(), dim=-1).type(attn.dtype)
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v).reshape(b, c, d, h, w)
        return x + self.proj_out(out)

# -------------------------------------------------------------------------
# Part 3: 主模型 UNetModel3D (全门控修正版)
# -------------------------------------------------------------------------

class UNetModel3D(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions, channel_mult=(1, 2, 4, 8), use_checkpoint=True, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.use_checkpoint = use_checkpoint

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(nn.Linear(model_channels, time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim))

        # --- 1. Down Path (Encoder) ---
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(3, in_channels, ch, 3, padding=1))])
        input_block_chans = [ch] # 记录每一层输出通道，供 Skip Connection 使用
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, time_embed_dim, 0, out_channels=int(mult * model_channels), use_checkpoint=use_checkpoint, use_scale_shift_norm=True)]
                ch = int(mult * model_channels)
                if ds in attention_resolutions: layers.append(AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, True)))
                input_block_chans.append(ch)
                ds *= 2

        # --- 2. Middle Block ---
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, 0, use_checkpoint=use_checkpoint, use_scale_shift_norm=True),
            AttentionBlock(ch, num_heads=num_heads, use_checkpoint=use_checkpoint),
            ResBlock(ch, time_embed_dim, 0, use_checkpoint=use_checkpoint, use_scale_shift_norm=True)
        )

        # --- 3. Up Path (Decoder) with Full Gating ---
        self.output_blocks = nn.ModuleList([])
        self.attention_gates = nn.ModuleList([]) # 初始化门控列表
        
        curr_ch = ch
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                # 获取对应的跳跃连接通道数
                ich = input_block_chans.pop()
                
                # === 核心修正：为每一个 Skip Connection 创建一个 Gate ===
                # F_g: 当前 Decoder 的特征通道 (curr_ch)
                # F_l: Skip Connection 的特征通道 (ich)
                # F_int: 中间层通道数，通常取 ich 的一半
                self.attention_gates.append(AttentionGate3D(F_g=curr_ch, F_l=ich, F_int=ich // 2))
                # ===================================================

                layers = [ResBlock(curr_ch + ich, time_embed_dim, 0, out_channels=int(model_channels * mult), use_checkpoint=use_checkpoint, use_scale_shift_norm=True)]
                curr_ch = int(model_channels * mult)
                if ds in attention_resolutions: layers.append(AttentionBlock(curr_ch, num_heads=num_heads, use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    layers.append(Upsample(curr_ch, True))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(normalization(curr_ch), nn.SiLU(), zero_module(conv_nd(3, curr_ch, out_channels, 3, padding=1)))

    def forward(self, x, timesteps, cond=None):
        if cond is not None: x = torch.cat([x, cond], dim=1)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        hs = []
        h = x.type(torch.float32)
        # Down Path
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        
        # Middle
        h = self.middle_block(h, emb)

        # Up Path with Full Gating
        for i, module in enumerate(self.output_blocks):
            skip_feat = hs.pop()
            
            # === 核心修正：无条件应用对应的 Gate ===
            # 因为我们在 __init__ 中为每个 output_block 都创建了 gate
            # 这里的 i 索引直接对应 self.attention_gates[i]
            skip_feat = self.attention_gates[i](h, skip_feat)
            # =====================================
            
            h = torch.cat([h, skip_feat], dim=1)
            h = module(h, emb)
            
        return self.out(h)

# -------------------------------------------------------------------------
# Part 4: 包装类 (对接 I2SB 训练脚本)
# -------------------------------------------------------------------------

class Image3DNet(nn.Module):
    def __init__(self, use_fp16=False, use_attention=True):
        super().__init__()
        attn_res = (8,) if use_attention else ()
        # 注意：这里 num_res_blocks=2，channel_mult=(1, 2, 4, 8) 是经典配置
        self.diffusion_model = UNetModel3D(
            in_channels=2, model_channels=64, out_channels=1, 
            num_res_blocks=2, attention_resolutions=attn_res,
            channel_mult=(1, 2, 4, 8)
        )
        if use_fp16: self.diffusion_model.half()

    def forward(self, x, steps, cond=None):
        return self.diffusion_model(x, steps, cond=cond)