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
    def __init__(self, in_channels=1, num_classes=2, growth_rate=32,
                 block_config=(6, 12, 24, 16), init_features=64, bn_size=4, drop_rate=0.0):
    # def __init__(self, in_channels=1, num_classes=2, growth_rate=32,
    #              block_config=(6, 12), init_features=64, bn_size=4, drop_rate=0.0):
        super(DenseNet3D, self).__init__()

        # 初始卷积层
        self.features = nn.Sequential(
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
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(in_channels=num_features, out_channels=num_features // 2)
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = num_features // 2

        # 最后的 BatchNorm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))

        # 分类器
        self.classifier1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 128)
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
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # x = x[:, 0:1, :, :, :]
        x = self.features(x)
        features = self.classifier1(x)
        x = self.classifier(features)
        return x, features

if __name__ == '__main__':
    model = DenseNet3D(num_classes=2, in_channels=2)
    model.eval()
    x = torch.randn(1, 2, 128, 128, 128)  # batch=1, channel=2
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    
    # 先保存为PyTorch格式
    torch.save(model.state_dict(), "densenet3d_model.pth")
    print("模型已保存为PyTorch格式: densenet3d_model.pth")
    
    # 导出为ONNX格式
    try:
        torch.onnx.export(
            model, 
            x, 
            "densenet3d_model.onnx",
            input_names=['input'],
            output_names=['logits', 'features'],
            opset_version=18,  # 使用最新的opset版本以避免版本转换问题
            do_constant_folding=True,
            verbose=False
        )
        print("模型已成功导出为ONNX格式: densenet3d_model.onnx")
    except ModuleNotFoundError:
        print("缺少onnxscript依赖，请运行: pip install onnxscript onnx")
    except Exception as e:
        print(f"导出ONNX时出错: {e}")