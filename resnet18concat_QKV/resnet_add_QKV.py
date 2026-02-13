import torch
import torch.nn as nn
import torch.nn.functional as F
from QKV import CrossModalAttention3D

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

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
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


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=400, input_channels=3, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(2, 2, 2),
                               padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cbam = CBAM3D(512)


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
                nn.BatchNorm3d(planes * block.expansion),
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
        x = self.cbam(x)+x
        # 4 4 4

        # x = self.avgpool(x)   # [B, 512, 1, 1, 1]
        # x = torch.flatten(x, 1)
        # x = self.dropout(x)  # 保存dropout后的特征用于可视化
        # features = self.fc(x)
        # x = self.classifier(features)
        return x  # 返回分类输出和特征向量
class dual_ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1, dropout_prob=0.3):
        super(dual_ResNet3D, self).__init__()
        self.res1 = ResNet3D(block, layers, num_classes, input_channels, dropout_prob)
        self.res2 = ResNet3D(block, layers, num_classes, input_channels, dropout_prob)
        self.QKV=CrossModalAttention3D(512)
        self.conv_final = nn.Conv3d(512, 128, kernel_size=1, stride=1, padding=1)
        self.bn_final = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.45)
        self.fc = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x1= x[:,0:1,:,:,:]
        x2= x[:,1:2,:,:,:]
        out1 = self.res1(x1)
        out2 = self.res2(x2)
        out1, out2 = self.QKV(out1, out2)
        out = out1 + out2  # 初步融合
        out = self.conv_final(out)
        out = self.bn_final(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        features = self.fc(out)
        out = self.classifier(features)
        return out, features

def resnet18_3d(num_classes=2, input_channels=1, dropout_prob=0.3):
    return dual_ResNet3D(BasicBlock3D, [2, 2, 2, 2],
                    num_classes=num_classes,
                    input_channels=input_channels,
                    dropout_prob=dropout_prob)
if __name__ == '__main__':
    model = resnet18_3d(num_classes=2, input_channels=2)
    x = torch.randn(2, 2, 128,128,128)  # batch=2, channel=2
    output, features = model(x)
    print(output.shape)  # 输出: [2, 2]
    print(features.shape)  # 输出: [2, 128]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")  
    print("ResNet3D模型测试通过！")
#     print(model)  # 输出完整架构
#     torch.onnx.export(model, x, "model.onnx")
# # 然后用 netron 打开：https://netron.app/
