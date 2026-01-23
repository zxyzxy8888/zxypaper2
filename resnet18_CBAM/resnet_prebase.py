import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# 1. CBAM 3D模块定义
# -------------------------
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, planes)

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


class ResNet3D_CBAM(nn.Module):
    def __init__(self, block=BasicBlock3D, layers=[2, 2, 2, 2], num_classes=2, input_channels=1, dropout_prob=0.5):
        super(ResNet3D_CBAM, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(1, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, 128)
        self.classifier = nn.Linear(128, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)     # [B, C, T, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 输出: [B, 512, 16, 4, 4] 对应输入 128^3
        return x 

class MultiModalResNet_Concat(nn.Module):
    def __init__(self, num_classes=2, dropout_prob=0.5):
        super().__init__()
        self.mri_encoder = ResNet3D_CBAM(num_classes=2, dropout_prob=0.0)
        self.pet_encoder = ResNet3D_CBAM(num_classes=2, dropout_prob=0.0)
        
        # 融合层: 1024 channels -> 512
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.Identity()
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(512, 128)
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        fm = self.mri_encoder(x[:, 0:1, :, :, :])  # (B,512,16,4,4)
        fp = self.pet_encoder(x[:, 1:2, :, :, :])  # (B,512,16,4,4)
        
        # 通道拼接
        f_concat = torch.cat([fm, fp], dim=1)  # (B,1024,16,4,4)
        f_fused = self.fusion_conv(f_concat)   # (B,512,16,4,4)
        
        f_pooled = self.avgpool(f_fused)       # (B,512,1,1,1)
        f_pooled = torch.flatten(f_pooled, 1)  # (B,512)
        
        f_pooled = self.dropout(f_pooled)
        features = self.fc(f_pooled)           # (B,128)
        out = self.classifier(features)        # (B,2)
        return out, features
# -------------------------
# 5. 测试模型结构
# -------------------------
if __name__ == "__main__":
    model = MultiModalResNet_Concat(num_classes=2, dropout_prob=0.5)
    x = torch.randn(2, 2, 128, 128, 128)  # 3D 输入 (B, C, D, H, W)
    out = model(x)
    # print(model)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")