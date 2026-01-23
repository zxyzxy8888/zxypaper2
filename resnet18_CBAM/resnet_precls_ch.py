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

class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.Hardswish()

        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, bias=True)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, bias=True)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, bias=True)

    def forward(self, x):
        n, c, d, h, w = x.size()

        x_d = x.mean(dim=(3,4), keepdim=True)          # [N,C,D,1,1]
        x_h = x.mean(dim=(2,4), keepdim=True).permute(0,1,3,2,4)  # [N,C,H,1,1]
        x_w = x.mean(dim=(2,3), keepdim=True).permute(0,1,4,2,3)  # [N,C,W,1,1]

        y = torch.cat([x_d, x_h, x_w], dim=2)          # [N,C,D+H+W,1,1]
        y = self.act(self.bn1(self.conv1(y)))

        y_d, y_h, y_w = torch.split(y, [d, h, w], dim=2)

        a_d = torch.sigmoid(self.conv_d(y_d))                              # [N,oup,D,1,1]
        a_h = torch.sigmoid(self.conv_h(y_h)).permute(0,1,3,2,4)           # [N,oup,1,H,1]
        a_w = torch.sigmoid(self.conv_w(y_w)).permute(0,1,3,4,2)           # [N,oup,1,1,W]

        return x * a_d * a_h * a_w
# -------------------------
# 2. BasicBlock3D
# -------------------------
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, att=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.att = att

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.att is not None:
            out = self.att(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out


# -------------------------
# 3. ResNet3D-18 Backbone + CBAM（输出512维特征）
# -------------------------
class ResNet3D_CBAM_Backbone(nn.Module):
    def __init__(self, block, layers, in_channels=1, dropout_prob=0.5, layer4_att_factory=None):
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_factory=layer4_att_factory)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1, att_factory=None):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )

        att = att_factory(planes) if att_factory is not None else None
        layers = [block(self.in_planes, planes, stride, downsample, att=att)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            att = att_factory(planes) if att_factory is not None else None
            layers.append(block(self.in_planes, planes, att=att))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # # 更标准的写法：残差式CBAM
        # x = x + self.cbam(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)   # [B, 512]
        # x = self.dropout(x)
        return x


# -------------------------
# 4. 通道级门控融合（更细粒度）
# -------------------------
class ChannelGatedFusion(nn.Module):
    """
    输出 alpha: [B, F]，逐通道决定MRI/PET占比
    fused = alpha*mri + (1-alpha)*pet
    """
    def __init__(self, feature_dim, hidden_ratio=0.5):
        super().__init__()
        hidden = max(64, int(feature_dim * hidden_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, feat_mri, feat_pet):
        alpha = self.mlp(torch.cat([feat_mri, feat_pet], dim=1))  # [B, F] in (0,1)
        fused = alpha * feat_mri + (1.0 - alpha) * feat_pet
        return fused

from QKV import CrossModalAttention3D
# -------------------------
# 5. 双分支 + 通道门控融合 + 分类
# -------------------------
class ResNet3D_CBAM_Fusion(nn.Module):
    def __init__(self, block, layers, num_classes=2, dropout_prob=0.3):
        super().__init__()
        self.backbone_mri = ResNet3D_CBAM_Backbone(
            block, layers, in_channels=1, dropout_prob=dropout_prob,
            layer4_att_factory=lambda c: CoordAtt3D(c, c)
        )
        self.backbone_pet = ResNet3D_CBAM_Backbone(
            block, layers, in_channels=1, dropout_prob=dropout_prob,
            layer4_att_factory=lambda c: CBAM3D(c)
        )
        self.bicross_atten = CrossModalAttention3D()
        self.postprocess = nn.Sequential(nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten())
        self.fusion = ChannelGatedFusion(feature_dim=512, hidden_ratio=0.5)

        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: [B, 2, D, H, W]
        x_mri = x[:, 0:1]
        x_pet = x[:, 1:2]

        feat_mri = self.backbone_mri(x_mri)  # [B, 512]
        feat_pet = self.backbone_pet(x_pet)  # [B, 512]
        feat_mri, feat_pet = self.bicross_atten(feat_mri, feat_pet)
        feat_mri = self.postprocess(feat_mri)
        feat_pet = self.postprocess(feat_pet)

        fused_feat = self.fusion(feat_mri, feat_pet)  # [B, 512], [B, 512]

        fused_feat = self.dropout(fused_feat)
        out = self.classifier(fused_feat)

        return out, fused_feat


def resnet3d18_cbam(num_classes=2, dropout_prob=0.5):
    return ResNet3D_CBAM_Fusion(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes, dropout_prob=dropout_prob)


if __name__ == "__main__":
    model = resnet3d18_cbam(num_classes=2, dropout_prob=0.5)
    x = torch.randn(2, 2, 64, 128, 128)
    out, features, alpha = model(x)
    print("输出 shape:", out.shape)
    print("特征 shape:", features.shape)
    print("alpha shape:", alpha.shape)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")