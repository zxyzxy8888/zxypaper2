import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. CBAM 3D模块定义
# -------------------------
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        hidden = max(1, in_planes // reduction)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)


class CBAM3D(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention3D(in_planes, reduction)
        self.sa = SpatialAttention3D(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, dropout_prob=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout_prob)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out

# -------------------------
# 3. ResNet3D-18 Backbone + CBAM（输出512维特征）
# -------------------------
class ResNet3D_CBAM_Backbone(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super().__init__()
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cbam = CBAM3D(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 更标准的写法：残差式CBAM
        x = x + self.cbam(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)   # [B, 512]
        x = self.dropout(x)
        return x


# -------------------------
# 4. 三路通道级门控融合（MRI / PET / inter）
# -------------------------
import torch
import torch.nn as nn

class ChannelGatedFusion(nn.Module):
    def __init__(self, feature_dim, hidden_ratio=0.5, dropout=0.2):
        super().__init__()
        hidden = max(64, int(feature_dim * hidden_ratio))
        self.mri_norm = nn.LayerNorm(feature_dim)
        self.pet_norm = nn.LayerNorm(feature_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, feature_dim * 3)  # [B, 3F] logits
        )
    def forward(self, feat_mri, feat_pet):
        # 保留 raw
        m_raw = feat_mri
        p_raw = feat_pet
        inter_raw = m_raw * p_raw  # 用于算 w 的交互项
        # 用 norm 特征算权重（更稳定）
        m = self.mri_norm(m_raw)
        p = self.pet_norm(p_raw)
        inter_gate = m * p  # 用于算 w 的交互项
        B, F = m.shape
        logits = self.mlp(torch.cat([m, p, inter_gate], dim=1)).reshape(B, 3, F)
        w = torch.softmax(logits, dim=1)
        fused = w[:, 0, :] * m_raw + w[:, 1, :] * p_raw + w[:, 2, :] * inter_raw
        return fused

# -------------------------
# 5. 双分支 + 通道门控融合 + 分类
# -------------------------
class ResNet3D_CBAM_Fusion(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_prob=0.3):
        super().__init__()
        self.backbone_mri = ResNet3D_CBAM_Backbone(block, layers, in_channels=1)
        self.backbone_pet = ResNet3D_CBAM_Backbone(block, layers, in_channels=1)

        self.fusion = ChannelGatedFusion(feature_dim=512, hidden_ratio=0.5)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
        )
        self.classifier_out = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: [B, 2, D, H, W]
        x_mri = x[:, 0:1]
        x_pet = x[:, 1:2]

        feat_mri = self.backbone_mri(x_mri)  # [B, 512]
        feat_pet = self.backbone_pet(x_pet)  # [B, 512]

        fused_feat = self.fusion(feat_mri, feat_pet)  # [B, 512], [B, 512]
        features= self.classifier(fused_feat)
        out = self.classifier_out(features)
        return out, features

def resnet3d18_cbam_ch_gate(num_classes=2, dropout_prob=0.3):
    return ResNet3D_CBAM_Fusion(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, dropout_prob=dropout_prob)

if __name__ == "__main__":
    model = resnet3d18_cbam_ch_gate(num_classes=2, dropout_prob=0.3)
    x = torch.randn(2, 2, 64, 128, 128)
    out, features = model(x)
    print("输出 shape:", out.shape)
    print("特征 shape:", features.shape)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")