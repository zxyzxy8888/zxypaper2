import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, 256)
        self.classifier = nn.Linear(256, num_classes)

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
        # x=x[:, 1:2, ...]  # 只取第er个通道作为输入
        x = self.conv1(x)     # [B, C, T, H, W]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 4 4 4
        x = self.avgpool(x)   # [B, 512, 1, 1, 1]
        x = torch.flatten(x, 1)
        x = self.dropout(x)  # 保存dropout后的特征用于可视化
        features = self.fc(x)
        x = self.classifier(features)
        return x, features  # 返回分类输出和特征向量


def resnet18_3d(num_classes=2, input_channels=3, dropout_prob=0.5):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2],
                    num_classes=num_classes,
                    input_channels=input_channels,
                    dropout_prob=dropout_prob)
if __name__ == '__main__':
    model = resnet18_3d(num_classes=2, input_channels=2)
    x = torch.randn(2, 2, 128,128,128)  # batch=2, channel=2
    output, features = model(x)
    print(output.shape)  # 输出: [2, 2]
    print(features.shape)  # 输出: [2, 256]
#     print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")  
#     print("ResNet3D模型测试通过！")
# #     print(model)  # 输出完整架构
# #     torch.onnx.export(model, x, "model.onnx")
# # # 然后用 netron 打开：https://netron.app/
