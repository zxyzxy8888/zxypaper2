import math
import torch
import torch.nn as nn


# -------------------------
# helpers
# -------------------------
def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def drop_connect(x, drop_prob: float, training: bool):
    """Stochastic depth for residual branch (3D)."""
    if (not training) or drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    rand = keep_prob + torch.rand((x.shape[0], 1, 1, 1, 1), device=x.device, dtype=x.dtype)
    binary = torch.floor(rand)
    return x / keep_prob * binary


class ConvBNAct3D(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, groups=1, act=True):
        padding = kernel_size // 2
        layers = [
            nn.Conv3d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_c),
        ]
        if act:
            layers.append(SiLU())
        super().__init__(*layers)


class SqueezeExcite3D(nn.Module):
    def __init__(self, in_c, se_ratio=0.25):
        super().__init__()
        se_c = max(1, int(in_c * se_ratio))
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Conv3d(in_c, se_c, kernel_size=1)
        self.act = SiLU()
        self.fc2 = nn.Conv3d(se_c, in_c, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        s = self.pool(x)
        s = self.fc1(s)
        s = self.act(s)
        s = self.fc2(s)
        s = self.gate(s)
        return x * s


class MBConv3D(nn.Module):
    """
    EfficientNet MBConv block (3D):
    expand -> depthwise -> SE -> project, residual if possible.
    """
    def __init__(self, in_c, out_c, stride, expand_ratio, kernel_size=3, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        assert stride in (1, 2)
        self.drop_rate = drop_rate

        hidden_c = int(round(in_c * expand_ratio))
        self.use_res = (stride == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct3D(in_c, hidden_c, kernel_size=1, stride=1, act=True))

        layers.append(
            ConvBNAct3D(hidden_c, hidden_c, kernel_size=kernel_size, stride=stride, groups=hidden_c, act=True)
        )

        self.conv = nn.Sequential(*layers)
        self.se = SqueezeExcite3D(hidden_c, se_ratio=se_ratio) if se_ratio and se_ratio > 0 else nn.Identity()

        self.project = nn.Sequential(
            nn.Conv3d(hidden_c, out_c, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(out_c),
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        out = self.project(out)
        if self.use_res:
            out = drop_connect(out, self.drop_rate, self.training)
            out = out + x
        return out


class EfficientNet3DBackbone(nn.Module):
    """
    EfficientNet-B0-like backbone (3D). 输出特征图 [B, C, d, h, w]（不做全局池化/分类）
    """
    def __init__(
        self,
        in_channels=1,
        width_mult=1.0,
        depth_mult=1.0,
        drop_connect_rate=0.2,
    ):
        super().__init__()

        # B0 baseline: (expand, out_c, repeats, stride, kernel)
        base_cfg = [
            (1,  16, 1, 1, 3),
            (6,  24, 2, 2, 3),
            (6,  40, 2, 2, 5),
            (6,  80, 3, 2, 3),
            (6, 112, 3, 1, 5),
            (6, 192, 4, 2, 5),
            (6, 320, 1, 1, 3),
        ]

        def round_channels(c):
            return _make_divisible(c * width_mult, 8)

        def round_repeats(r):
            return int(math.ceil(r * depth_mult))

        stem_out = round_channels(32)
        self.stem = ConvBNAct3D(in_channels, stem_out, kernel_size=3, stride=2, act=True)

        total_blocks = sum(round_repeats(r) for (_, _, r, _, _) in base_cfg)
        block_id = 0

        blocks = []
        in_c = stem_out
        for expand, c, r, s, k in base_cfg:
            out_c = round_channels(c)
            repeats = round_repeats(r)
            for i in range(repeats):
                stride = s if i == 0 else 1
                dc = drop_connect_rate * block_id / max(1, total_blocks - 1)  # linear schedule
                blocks.append(MBConv3D(in_c, out_c, stride=stride, expand_ratio=expand, kernel_size=k,
                                      se_ratio=0.25, drop_rate=dc))
                in_c = out_c
                block_id += 1

        self.blocks = nn.Sequential(*blocks)

        head_out = round_channels(1280)
        self.head = ConvBNAct3D(in_c, head_out, kernel_size=1, stride=1, act=True)
        self.out_channels = head_out

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


class EfficientNet3D_TwoBranchConcatBeforePool(nn.Module):
    """
    输入 x: [B, 2, D, H, W]
    两个独立 EfficientNet3D backbone 分别处理两个通道 (x[:,0], x[:,1])
    在特征图层面 concat(通道维) -> 再全局池化 -> MLP -> 分类
    """
    def __init__(
        self,
        num_classes=2,
        width_mult=1.0,
        depth_mult=1.0,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
        embed_dim=128,   # 输出 features 维度（可改）
    ):
        super().__init__()

        # 两个分支独立参数
        self.backbone_a = EfficientNet3DBackbone(
            in_channels=1, width_mult=width_mult, depth_mult=depth_mult, drop_connect_rate=drop_connect_rate
        )
        self.backbone_b = EfficientNet3DBackbone(
            in_channels=1, width_mult=width_mult, depth_mult=depth_mult, drop_connect_rate=drop_connect_rate
        )

        concat_c = self.backbone_a.out_channels + self.backbone_b.out_channels  # 2 * C

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(concat_c, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, embed_dim),
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: [B, 2, D, H, W]
        return:
          logits:   [B, num_classes]
          features: [B, embed_dim]
        """
        if x.dim() != 5 or x.size(1) != 2:
            raise ValueError(f"Expected x with shape [B, 2, D, H, W], got {tuple(x.shape)}")

        xa = x[:, 0:1, ...]
        xb = x[:, 1:2, ...]

        fa = self.backbone_a(xa)  # [B,C,d,h,w]
        fb = self.backbone_b(xb)  # [B,C,d,h,w]

        f = torch.cat([fa, fb], dim=1)  # 先 concat

        f = self.avgpool(f)             # 再池化
        f = torch.flatten(f, 1)         # [B, 2C]

        features = self.mlp(f)          # [B, embed_dim]
        logits = self.classifier(features)
        return logits, features


# -------------------------
# quick test
# -------------------------
if __name__ == "__main__":
    model = EfficientNet3D_TwoBranchConcatBeforePool(num_classes=2, width_mult=1.0, depth_mult=1.0)
    x = torch.randn(2, 2, 64, 128, 128)
    y, feat = model(x)
    print(y.shape, feat.shape)  # torch.Size([2, 2]) torch.Size([2, 128])