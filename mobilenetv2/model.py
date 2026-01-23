import torch
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU3D(nn.Sequential):
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv3d(in_c, out_c, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual3D(nn.Module):
    """
    MobileNetV2 inverted residual block (3D version):
    expand (1x1x1) -> depthwise (3x3x3) -> project (1x1x1)
    """
    def __init__(self, in_c, out_c, stride, expand_ratio):
        super().__init__()
        assert stride in (1, 2)

        hidden_dim = int(round(in_c * expand_ratio))
        self.use_res_connect = (stride == 1 and in_c == out_c)

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU3D(in_c, hidden_dim, kernel_size=1, stride=1))
        # dw
        layers.append(ConvBNReLU3D(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        # pw-linear
        layers.append(nn.Conv3d(hidden_dim, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm3d(out_c))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2_3D_Backbone(nn.Module):
    """
    输出一个 3D 特征图 (B, last_channel, d, h, w)，不做全局池化/分类。
    """
    def __init__(self, in_channels=1, width_mult=1.0, round_nearest=8):
        super().__init__()

        # MobileNetV2 配置: (t, c, n, s)
        cfgs = [
            # expand, out, num_blocks, stride
            (1,  16, 1, 1),
            (6,  24, 2, 2),
            (6,  32, 3, 2),
            (6,  64, 4, 2),
            (6,  96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        input_channel = _make_divisible(32 * width_mult, round_nearest)
        self.stem = ConvBNReLU3D(in_channels, input_channel, kernel_size=3, stride=2)

        layers = []
        for t, c, n, s in cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual3D(input_channel, output_channel, stride=stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)

        last_channel = _make_divisible(1280 * width_mult, round_nearest) if width_mult > 1.0 else 1280
        self.head = ConvBNReLU3D(input_channel, last_channel, kernel_size=1, stride=1)

        self.out_channels = last_channel

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x


class MobileNetV2_3D_TwoBranchConcatBeforePool(nn.Module):
    """
    x: [B, 2, D, H, W]
    两分支 MobileNetV2-3D -> concat(通道维) -> AdaptiveAvgPool3d(1) -> FC
    """
    def __init__(self, num_classes=2, width_mult=1.0):
        super().__init__()

        self.backbone_a = MobileNetV2_3D_Backbone(in_channels=1, width_mult=width_mult)
        self.backbone_b = MobileNetV2_3D_Backbone(in_channels=1, width_mult=width_mult)

        concat_channels = self.backbone_a.out_channels + self.backbone_b.out_channels  # 2 * out_channels

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier1 = nn.Sequential(
            nn.Linear(concat_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        if x.dim() != 5 or x.size(1) != 2:
            raise ValueError(f"Expected x with shape [B, 2, D, H, W], got {tuple(x.shape)}")

        xa = x[:, 0:1, ...]
        xb = x[:, 1:2, ...]

        fa = self.backbone_a(xa)  # [B,C,d,h,w]
        fb = self.backbone_b(xb)  # [B,C,d,h,w]

        f = torch.cat([fa, fb], dim=1)  # 先 concat

        f = self.avgpool(f)             # 再池化
        f = torch.flatten(f, 1)         # [B, 2C]

        features = self.classifier1(f)  # [B,128]
        logits = self.classifier(features)

        return logits, features