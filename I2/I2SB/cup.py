import torch
from thop import profile, clever_format

# 线性注意力是当前 UNet_linearatt 的默认全局注意力；AttentionBlock 是传统的全局自注意力（O(N^2)）。
from UNet_linearatt import LinearAttentionBlock3D
from UNet import AttentionBlock, QKVAttention


def _count_linear_attn_ops(module, inputs, outputs):
    """Estimate FLOPs for LinearAttentionBlock3D's two einsums (context & apply)."""
    x = inputs[0]
    b, c, d, h, w = x.shape
    n = d * h * w
    heads = module.num_heads
    head_dim = c // heads
    # 两次矩阵乘：context 和 apply，均为 b * heads * head_dim * head_dim * n
    ops = 2 * b * heads * head_dim * head_dim * n
    module.total_ops += torch.DoubleTensor([ops])

def _profile(attn_cls, name, C, D, H, W, **attn_kwargs):
    """Profile 单个注意力模块的 MACs / Params / 显存。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("请在 GPU 环境下运行以获取准确的显存占用 (Mem.) 数据。")
        return

    attn = attn_cls(**attn_kwargs).to(device)
    dummy_x = torch.randn(1, C, D, H, W, device=device)

    # 自定义 FLOPs：传统 QKVAttention 计数，线性注意力额外注册
    custom_ops = {QKVAttention: QKVAttention.count_flops}
    if attn_cls is LinearAttentionBlock3D:
        custom_ops[LinearAttentionBlock3D] = _count_linear_attn_ops

    macs, params = profile(attn, inputs=(dummy_x,), verbose=False, custom_ops=custom_ops)
    macs_str, params_str = clever_format([macs, params], "%.2f")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    attn.train()
    out = attn(dummy_x)
    out.sum().backward()  # 反传以触发激活显存
    mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f" └─ {name} -> Comp(MACs): {macs_str} | Params: {params_str} | Mem: {mem_mb:.2f} MB")


def profile_isolated_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("请在 GPU 环境下运行以获取准确的显存占用 (Mem.) 数据。")
        return

    print("=== 开始精准测算单独的注意力模块 (Attention Only) ===")

    # -----------------------------------------------------------------
    # 情景 1: 分辨率 24x24x24 (对应 Level 2, ds=4)
    # 通道数 C = model_channels(64) * mult(4) = 256
    # -----------------------------------------------------------------
    C_24, D_24, H_24, W_24 = 256, 24, 24, 24
    print(f"\n[测试 1] 特征图尺寸: 24x24x24, 通道数: {C_24}")
    
    _profile(
        LinearAttentionBlock3D,
        name="LinearAttentionBlock3D (默认，O(N))",
        C=C_24,
        D=D_24,
        H=H_24,
        W=W_24,
        channels=C_24,
        num_heads=4,
        use_checkpoint=False,
    )

    _profile(
        AttentionBlock,
        name="AttentionBlock (传统 QKV, O(N^2))",
        C=C_24,
        D=D_24,
        H=H_24,
        W=W_24,
        channels=C_24,
        num_heads=4,
        use_checkpoint=False,
        use_new_attention_order=True,
    )


    # -----------------------------------------------------------------
    # 情景 2: 分辨率 12x12x12 (对应 Level 3 & Middle Block, ds=8)
    # 通道数 C = model_channels(64) * mult(8) = 512
    # -----------------------------------------------------------------
    C_12, D_12, H_12, W_12 = 512, 12, 12, 12
    print(f"\n[测试 2] 特征图尺寸: 12x12x12, 通道数: {C_12}")
    
    _profile(
        LinearAttentionBlock3D,
        name="LinearAttentionBlock3D (默认，O(N))",
        C=C_12,
        D=D_12,
        H=H_12,
        W=W_12,
        channels=C_12,
        num_heads=4,
        use_checkpoint=False,
    )

    _profile(
        AttentionBlock,
        name="AttentionBlock (传统 QKV, O(N^2))",
        C=C_12,
        D=D_12,
        H=H_12,
        W=W_12,
        channels=C_12,
        num_heads=4,
        use_checkpoint=False,
        use_new_attention_order=True,
    )
    
    print("\n注：LinearAttentionBlock3D 是当前 UNet_linearatt 的默认全局注意力；以上对比给出了线性与传统 QKV 自注意力的计算量 / 参数量 / 显存差异。")

if __name__ == "__main__":
    profile_isolated_attention()