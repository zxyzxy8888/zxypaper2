import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.expansion = 1

        # self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=(2, 2, 2),
        #                        padding=(3, 3, 3), bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        # self.relu = nn.ReLU(inplace=True)

        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(2, 2, 2), padding=1)
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # Dense Blocks + Transitions
        
        num_features = 64
        block_config=(6, 12)
        growth_rate=32
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_channels=num_features,
                                growth_rate=growth_rate, bn_size=4, drop_rate=0)
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 最后的 BatchNorm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        self.conv1x1 = nn.Conv3d(num_features, 128, kernel_size=1, stride=1, bias=False)
        self.in_planes = 128  # 更新in_planes为conv1x1的输出通道
        self.layer3 = self._make_layer(BasicBlock3D, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock3D, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, 256)
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
        x=x[:, 0:1, ...]  # 只取前两个通道作为输入
        # x = self.conv1(x)     # [B, C, T, H, W]
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        x= self.features(x)
        x=self.conv1x1(x)
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
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")  
#     print("ResNet3D模型测试通过！")
# #     print(model)  # 输出完整架构
# #     torch.onnx.export(model, x, "model.onnx")
# # # 然后用 netron 打开：https://netron.app/
