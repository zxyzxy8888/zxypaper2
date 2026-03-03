import torch as th
from thop import profile
# 假设你在 train.py 或 test.py 里测试
from UNet_linearatt import UNetModel, QKVAttention, QKVAttentionLegacy, LinearAttentionBlock3D

# 1. 实例化你的 3D UNet (分辨率 96, 开启 12 层的 EA)
model = UNetModel(
    image_size=96,
    in_channels=1,
    model_channels=64,
    out_channels=1,
    attention_resolutions=(12, ) # 这里填 12 或 24
)

# 2. 伪造一个输入的张量和时间步
dummy_x = th.randn(2, 1, 96, 96, 96) # Batch=1, Channels=2
dummy_t = th.randint(0, 1000, (2,))  # 时间步

# 3. 告诉 thop：遇到这几个特定的注意力类时，用它们自己的 count_flops 规则算
custom_ops = {
    QKVAttention: QKVAttention.count_flops,
    QKVAttentionLegacy: QKVAttentionLegacy.count_flops,
    LinearAttentionBlock3D: LinearAttentionBlock3D.count_flops, # <--- 你的 O(N) 法宝
}

# 4. 一键测算！
macs, params = profile(
    model, 
    inputs=(dummy_x, dummy_t), 
    custom_ops=custom_ops
)

print(f"参数量 (Params): {params / 1e6:.2f} M")
print(f"计算量 (MACs/FLOPs): {macs / 1e9:.2f} G")