import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- 新增模块：SE注意力机制 ----------------------
class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# ---------------------- 新增模块：改良的中间过渡层 ----------------------
class MiddleTransition(nn.Module):
    """
    用于连接 DenseNet 特征和 ResNet 层
    包含：BN -> ReLU -> SE Attention -> Conv3x3 (降维)
    """
    def __init__(self, in_planes, out_planes):
        super(MiddleTransition, self).__init__()
        self.bn = nn.BatchNorm3d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock3D(in_planes, reduction=16) # 先做注意力
        # 使用 3x3 卷积代替 1x1，保留更多空间信息
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.se(out) # 关键：筛选特征
        out = self.conv(out)
        return out

# ---------------------- 原有基础模块保持不变 ----------------------
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

# ---------------------- 修改后的主模型 ----------------------
class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=400, input_channels=3, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        
        # 初始层
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Blocks + Transitions
        num_features = 64
        block_config = (6, 12)
        growth_rate = 32
        
        for i, num_layers in enumerate(block_config):
            block_d = _DenseBlock(num_layers=num_layers, in_channels=num_features,
                                growth_rate=growth_rate, bn_size=4, drop_rate=0)
            self.features.add_module(f'denseblock{i + 1}', block_d)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # --- 修改点开始 ---
        # 此时 num_features 为 512
        # 我们不再使用简单的 conv1x1 压到 128，而是使用改良的 Transition 压到 256
        target_mid_planes = 256 
        
        # 替换原来的 self.conv1x1
        self.middle_transition = MiddleTransition(num_features, target_mid_planes)
        
        # 更新 self.in_planes 以匹配 layer3 的输入需求
        self.in_planes = target_mid_planes 
        
        # Layer 3: 输入 256, 输出 256, stride=2 (这里会进行下采样)
        self.layer3 = self._make_layer(BasicBlock3D, 256, layers[2], stride=2)
        
        # Layer 4: 输入 256, 输出 512, stride=2
        self.layer4 = self._make_layer(BasicBlock3D, 512, layers[3], stride=2)
        # --- 修改点结束 ---

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * BasicBlock3D.expansion, 256)
        self.classifier = nn.Linear(256, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 如果 stride!=1 或者 输入通道 != 输出通道，则需要 downsample
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x[:, 0:1, ...] 
        
        x = self.features(x)
        
        # 使用新的中间层
        x = self.middle_transition(x)
        
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        features = self.fc(x)
        x = self.classifier(features)
        return x, features

def resnet18_3d(num_classes=2, input_channels=3, dropout_prob=0.5):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2],
                    num_classes=num_classes,
                    input_channels=input_channels,
                    dropout_prob=dropout_prob)

if __name__ == '__main__':
    # 测试代码
    model = resnet18_3d(num_classes=2, input_channels=2)
    # 模拟输入：Batch=2, Channel=2, D=128, H=128, W=128
    x = torch.randn(2, 2, 128, 128, 128) 
    
    output, features = model(x)
    print(f"Output shape: {output.shape}")      # 预期: [2, 2]
    print(f"Features shape: {features.shape}")  # 预期: [2, 256]
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")