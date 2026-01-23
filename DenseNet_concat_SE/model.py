import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. SE 3D 模块定义
# -------------------------
class SE3D(nn.Module):
    """
    Squeeze-and-Excitation for 3D feature maps.
    x: (B, C, D, H, W)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))   # (B, C, 1, 1, 1)
        return x * w


# -------------------------
# 2. DenseNet 3D 基础模块
# -------------------------
class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm3d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(in_channels, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))

        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4, drop_rate=0.0):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                in_channels + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            self.add_module(f'denselayer{i + 1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm3d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(in_channels, out_channels,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


# -------------------------
# 3. 特征提取器：在 transition3 后插入 SE
# -------------------------
def make_features_with_se_after_transition3(
    in_channels: int,
    init_features: int,
    growth_rate: int,
    block_config=(6, 12, 24, 16),
    bn_size: int = 4,
    drop_rate: float = 0.0,
    se_reduction: int = 16,
):
    """
    DenseNet3D features，并在 transition3 后插入 SE3D。
    对于 block_config=(6,12,24,16)，transition3 位于 denseblock3 后。
    """
    features = nn.Sequential(
        nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm3d(init_features),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
    )

    num_features = init_features
    for i, num_layers in enumerate(block_config):
        # denseblock{i+1}
        block = _DenseBlock(
            num_layers=num_layers,
            in_channels=num_features,
            growth_rate=growth_rate,
            bn_size=bn_size,
            drop_rate=drop_rate
        )
        features.add_module(f'denseblock{i + 1}', block)
        num_features = num_features + num_layers * growth_rate

        # transition{i+1} (除了最后一个 block 后面没有 transition)
        if i != len(block_config) - 1:
            trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
            features.add_module(f'transition{i + 1}', trans)
            num_features = num_features // 2

            # 在 transition3 后插入 SE（即 i==2 时刚加完 transition3）
            if (i + 1) == 3:
                features.add_module('se_after_transition3', SE3D(num_features, reduction=se_reduction))

    features.add_module('norm_final', nn.BatchNorm3d(num_features))
    return features, num_features


# -------------------------
# 4. 双模态 DenseNet3D：每个模态一个 SE@transition3
# -------------------------
class DenseNet3D_SE_T3(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=2,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        init_features=64,
        bn_size=4,
        drop_rate=0.0,
        se_reduction=16,
    ):
        super().__init__()

        assert in_channels % 2 == 0, "in_channels 必须能被2整除（MRI/PET各占一半通道）"

        self.features_mri, num_features_mri = make_features_with_se_after_transition3(
            in_channels=in_channels // 2,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            drop_rate=drop_rate,
            se_reduction=se_reduction,
        )

        self.features_pet, num_features_pet = make_features_with_se_after_transition3(
            in_channels=in_channels // 2,
            init_features=init_features,
            growth_rate=growth_rate,
            block_config=block_config,
            bn_size=bn_size,
            drop_rate=drop_rate,
            se_reduction=se_reduction,
        )

        # 两个分支结构一致时，这里应相等；写成 assert 防止你改配置后悄悄不一致
        assert num_features_mri == num_features_pet
        num_features = num_features_mri

        self.classifier1 = nn.Sequential(
            nn.Conv3d(num_features * 2, 128, kernel_size=1, stride=1, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(128, 128),
        )
        self.classifier = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mri = x[:, 0:1, :, :, :]
        pet = x[:, 1:2, :, :, :]

        mri_feat = self.features_mri(mri)
        pet_feat = self.features_pet(pet)

        fused = torch.cat([mri_feat, pet_feat], dim=1)
        features = self.classifier1(fused)
        logits = self.classifier(features)
        return logits, features


if __name__ == '__main__':
    model = DenseNet3D_SE_T3(num_classes=2, in_channels=2)
    x = torch.randn(2, 2, 128, 128, 128)
    out, feat = model(x)
    print(out.shape)   # torch.Size([2, 2])
    print(feat.shape)  # torch.Size([2, 128])
    print(f"参数量: {sum(p.numel() for p in model.parameters())}")