import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. CBAM 3D模块定义
# -------------------------
class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(
            nn.Conv3d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_planes // reduction, in_planes, 1, bias=False)
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

# -------------------------
# 2. BasicBlock3D + CBAM
# -------------------------
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        return out

# -------------------------
# 3. ResNet3D-18 + CBAM
# -------------------------
class ResNet3D_CBAM(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_prob=0.5):
        super().__init__()
        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, 128)
        self.classfier = nn.Linear(128, num_classes)
        self.cbam = CBAM3D(512 * block.expansion)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
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
        x = self.cbam(x)+x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return x

class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # 输出两个权重
            nn.Softmax(dim=1)           # 归一化权重
        )

    def forward(self, feat_mri, feat_pet):
        combined = torch.cat([feat_mri, feat_pet], dim=1)  # [B, 2F]
        weights = self.gate(combined)                      # [B, 2]
        w_mri = weights[:, 0].unsqueeze(1)                 # [B, 1]
        w_pet = weights[:, 1].unsqueeze(1)                 # [B, 1]
        fused = w_mri * feat_mri + w_pet * feat_pet        # [B, F]
        return fused

class ResNet3D_CBAM_Fusion(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_prob=0.5):
        super().__init__()
        self.resnet_mri = ResNet3D_CBAM(block, layers, num_classes, dropout_prob)
        self.resnet_pet = ResNet3D_CBAM(block, layers, num_classes, dropout_prob)
        self.gated_fusion = GatedFusion(feature_dim=512)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x_mri = x[:, 0:1, :, :, :]  # MRI 通道
        x_pet = x[:, 1:2, :, :, :]  # PET 通道
        feat_mri = self.resnet_mri(x_mri)  # 提取 MRI 特征
        feat_pet = self.resnet_pet(x_pet)  # 提取 PET 特征
        fused_feat = self.gated_fusion(feat_mri, feat_pet)  # 融合特征
        fused_feat = self.dropout(fused_feat)  # dropout
        out = self.classifier(fused_feat)  # 分类
        return out, fused_feat


#-------------------------
# 4. 构建接口函数
# -------------------------
def resnet3d18_cbam(num_classes=2, dropout_prob=0.5):
    return ResNet3D_CBAM_Fusion(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, dropout_prob=dropout_prob)

# -------------------------
# 5. 测试模型结构
# -------------------------
if __name__ == "__main__":
    model = resnet3d18_cbam(num_classes=2, dropout_prob=0.5)
    x = torch.randn(2, 2, 64, 128, 128)  # 3D 输入 (B, C=2, D, H, W) - 2个通道：MRI和PET
    out, features = model(x)
    print("输出 shape:", out.shape)
    print("特征 shape:", features.shape)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")