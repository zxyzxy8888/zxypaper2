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

class DenseNet3D(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, growth_rate=32,
                 block_config=(6, 12, 24, 16), init_features=64, bn_size=4, drop_rate=0.3):
        super(DenseNet3D, self).__init__()

        # 为MRI创建独立的特征提取器
        self.features_mri = self._make_features(in_channels // 2, init_features, growth_rate, 
                                                 block_config, bn_size, drop_rate)
        
        # 为PET创建独立的特征提取器
        self.features_pet = self._make_features(in_channels // 2, init_features, growth_rate, 
                                                 block_config, bn_size, drop_rate)
        
        # 计算最终的特征数量
        num_features = init_features
        for i, num_layers in enumerate(block_config):
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                num_features = num_features // 2

        # 分类器（先pool再concat）
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(num_features * 2, 128)
        self.classifier = nn.Linear(128, num_classes)
        self._initialize_weights()
    
    def _make_features(self, in_channels, init_features, growth_rate, block_config, bn_size, drop_rate):
        """创建特征提取网络"""
        features = nn.Sequential(
            nn.Conv3d(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense Blocks + Transitions
        num_features = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, in_channels=num_features,
                                growth_rate=growth_rate, bn_size=bn_size, drop_rate=drop_rate)
            features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
                features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 最后的 BatchNorm
        features.add_module('norm_final', nn.BatchNorm3d(num_features))
        return features
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        mri=x[:,0:1,:,:,:]
        pet=x[:,1:2,:,:,:]
        mri = self.features_mri(mri)
        pet = self.features_pet(pet)
        x = torch.cat((mri, pet), dim=1)
        
        # 先分别pooling
        x= self.avgpool(x)
        x = torch.flatten(x, 1)
        features = self.fc1(x)
        x = self.classifier(features)
        return x, features
if __name__ == '__main__':
    model = DenseNet3D(num_classes=2, in_channels=2)
    x = torch.randn(2, 2, 128,128,128)  # batch=2, channel=1
    output, features = model(x)
    print(output.shape)  # 输出: [2, 2]
    print(features.shape)  # 输出: [2, 128]
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")