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
        super(CoordAtt3D, self).__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.Hardswish()
        
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        
        x_d = self.pool_d(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)

        # D方向
        y_d = self.conv1(x_d)
        y_d = self.act(self.bn1(y_d))
        a_d = self.conv_d(y_d).sigmoid()

        # H方向
        y_h = self.conv1(x_h)
        y_h = self.act(self.bn1(y_h))
        a_h = self.conv_h(y_h).sigmoid()

        # W方向
        y_w = self.conv1(x_w)
        y_w = self.act(self.bn1(y_w))
        a_w = self.conv_w(y_w).sigmoid()

        out = identity * a_d * a_h * a_w
        return out
    
import math

class ECA3D(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        """
        ECA-Net: Efficient Channel Attention for 3D.
        
        Args:
            channels: 输入特征图的通道数 (C)
            gamma: 用于计算卷积核大小的超参数 (默认 2)
            b: 用于计算卷积核大小的超参数 (默认 1)
        """
        super(ECA3D, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1  # 保证 k 是奇数

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        y = self.avg_pool(x)
        # (B, C, 1, 1, 1) -> (B, C, 1) -> permute -> (B, 1, C)
        y = y.squeeze(-1).squeeze(-1).permute(0, 2, 1)

        y = self.conv(y)

        y = self.sigmoid(y).permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)

        return x * y.expand_as(x)
class SimAM3D(nn.Module):
    def __init__(self, e_lambda=1e-4):
        """
        SimAM: A Simple, Parameter-Free Attention Module for 3D.
        
        Args:
            e_lambda: 防止除零的一个极小值 (epsilon)
        """
        super(SimAM3D, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        # x shape: (Batch, Channel, Depth, Height, Width)
        b, c, d, h, w = x.size()
        
        # 1. 计算空间维度的像素数 (n = D * H * W)
        n = d * h * w - 1
        
        # 2. 计算每个通道的均值和方差
        # 为了计算简便，我们计算 (x - mu)^2 的形式
        # view: (B, C, D*H*W)
        x_minus_mu_square = (x - x.mean(dim=[2, 3, 4], keepdim=True)).pow(2)
        
        # 3. 计算能量函数中的分母部分
        # y = x_minus_mu_square / (4 * (variance + lambda)) + 0.5
        # variance = sum(x_minus_mu_square) / n
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3, 4], keepdim=True) / n + self.e_lambda)) + 0.5
        
        # 4. Sigmoid 激活得到注意力权重
        return x * self.activaton(y)
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, att_type='ca'):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = downsample

        self.att_type = att_type
        if att_type == 'ca':
            self.att = CoordAtt3D(planes, planes)
        elif att_type == 'cbam':
            self.att = CBAM3D(planes) # CBAM通常只需要通道数
        elif att_type == 'eca':
            self.att = ECA3D(planes)  # ECA也只需要通道数
        elif att_type == 'simam':
            self.att = SimAM3D()      # SimAM通常是无参的或自适应的
        elif att_type == 'none':
            self.att = nn.Identity()  # 基准 Baseline

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.att_type != 'none':
            out = self.att(out) 
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

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, att_type='none')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, att_type='simam')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, att_type='simam')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, att_type='none')

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(512 * block.expansion, 128)
        self.classifier = nn.Linear(128, num_classes)
        self.cbam4 = CBAM3D(512 * block.expansion)

        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1, att_type='ca'):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = [block(self.in_planes, planes, stride, downsample, att_type=att_type)]
        self.in_planes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, att_type=att_type))

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
        # x = self.cbam4(x)+x
        return x
class dual_ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=2, input_channels=1, dropout_prob=0.3):
        super(dual_ResNet3D, self).__init__()
        # 每个流只处理单通道数据，所以固定为1
        self.res1 = ResNet3D(block, layers, num_classes, input_channels=1, dropout_prob=dropout_prob)
        self.res2 = ResNet3D(block, layers, num_classes, input_channels=1, dropout_prob=dropout_prob)
        self.conv_final = nn.Conv3d(1024, 128, kernel_size=1, stride=1, padding=1)
        self.bn_final = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(128, 128)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x1= x[:,0:1,:,:,:]
        x2= x[:,1:2,:,:,:]
        out1 = self.res1(x1)
        out2 = self.res2(x2)
        out = torch.cat((out1, out2), dim=1)  # 在通道维度上拼接
        out = self.conv_final(out)
        out = self.bn_final(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        features = self.fc(out)
        out = self.classifier(features)
        return out, features

def resnet18_3d(num_classes=2, input_channels=2, dropout_prob=0.3):
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
