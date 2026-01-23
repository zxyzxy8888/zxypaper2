import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Multi-Scale Integration Block (MS Block)
# -----------------------------
class MSBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super(MSBlock, self).__init__()
        reduced_channels = channels // reduction

        # Step 1: 1x1x1 conv to compress
        self.compress = nn.Sequential(
            nn.Conv3d(channels, reduced_channels, kernel_size=1),
            nn.BatchNorm3d(reduced_channels),
            nn.ReLU(inplace=True)
        )

        # Step 2: multi-scale conv (kernel size = 3, dilation = 1/2/3)
        self.conv3 = nn.Sequential(
            nn.Conv3d(reduced_channels, channels, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(reduced_channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        self.conv7 = nn.Sequential(
            nn.Conv3d(reduced_channels, channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True)
        )

        # Attention weights (channel-wise)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.BatchNorm1d(channels // reduction),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Linear(channels // reduction, channels)
        self.fc5 = nn.Linear(channels // reduction, channels)
        self.fc7 = nn.Linear(channels // reduction, channels)

        self.shortcut = nn.Conv3d(channels, channels, kernel_size=1)

        self.activation = nn.LeakyReLU(0.01)

    def forward(self, x):
        compressed = self.compress(x)
        y3 = self.conv3(compressed)
        y5 = self.conv5(compressed)
        y7 = self.conv7(compressed)

        # Soft attention weights
        y_sum = y3 + y5 + y7
        u = self.global_pool(y_sum).view(x.size(0), -1)
        v = self.fc(u)

        a = self.fc3(v)
        b = self.fc5(v)
        c = self.fc7(v)

        weights = torch.softmax(torch.stack([a, b, c], dim=1), dim=1)  # [B, 3, C]

        a_w = weights[:, 0, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        b_w = weights[:, 1, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c_w = weights[:, 2, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        out = y3 * a_w + y5 * b_w + y7 * c_w
        out = self.activation(out + self.shortcut(x))

        return out
class AttentionBlock(nn.Module):
    """Improved Attention Block (Section III-B-1)"""
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        
        # Multiple 3x3x3 convolution kernels
        self.conv = nn.Conv3d(in_channels, in_channels, 
                             kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(in_channels)
        
        # Use Sigmoid instead of ReLU to squeeze attention to [0, 1]
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention mask M
        mask = self.conv(x)  # [B, C, D, H, W]
        mask = self.bn(mask)
        mask = self.sigmoid(mask)  # M(x,y,z) ∈ (0, 1)
        
        # Weighted output: H = F * M (Equation 6)
        out = x * mask
        
        return out
class ConvBlock(nn.Module):
    """Basic Convolution Block with Shortcut"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                     stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Shortcut branch (1x1x1 conv for dimension matching)
        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv(x)
        out = self.relu(out + identity)
        return out

class AMSNet(nn.Module):
    """
    Attention-based 3D Multi-Scale CNN (AMSNet)
    
    Paper: An Attention-Based 3D CNN With Multi-Scale Integration Block 
           for Alzheimer's Disease Classification (IEEE JBHI 2022)
    """
    def __init__(self, num_classes=2, in_channels=1, reduction_ratio=16):
        super(AMSNet, self).__init__()
        
        # ========== Stage 1: Initial feature extraction ==========
        self.conv1 = ConvBlock(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.ms_block1 = MSBlock(32)
        self.attention1 = AttentionBlock(32)
        
        # ========== Stage 2: Deeper feature extraction ==========
        self.conv2 = ConvBlock(32, 64, kernel_size=3, stride=2, padding=1)
        self.ms_block2 = MSBlock(64,)
        self.attention2 = AttentionBlock(64)
        
        # ========== Stage 3: High-level feature extraction ==========
        self.conv3 = ConvBlock(64, 128, kernel_size=3, stride=2, padding=1)
        self.ms_block3 = MSBlock(128)
        
        # ========== Stage 4: Final semantic features ==========
        self.conv4_1 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = ConvBlock(256, 256, kernel_size=3, stride=1, padding=1)
        
        # ========== Stage 5: Classification ==========
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x: [B, 1, 91, 109, 91] (MRI input)
        
        # Stage 1
        x = self.conv1(x)          # [B, 32, 46, 55, 46]
        x = self.ms_block1(x)      # Multi-scale integration
        x = self.attention1(x)     # Attention refinement
        
        # Stage 2
        x = self.conv2(x)          # [B, 64, 23, 28, 23]
        x = self.ms_block2(x)
        x = self.attention2(x)
        
        # Stage 3
        x = self.conv3(x)          # [B, 128, 12, 14, 12]
        x = self.ms_block3(x)
        
        # Stage 4
        x = self.conv4_1(x)        # [B, 256, 6, 7, 6]
        x = self.conv4_2(x)        # [B, 256, 6, 7, 6]
        
        # Stage 5: Classification
        x = self.global_avg_pool(x)  # [B, 256, 1, 1, 1]
        x = torch.flatten(x, 1)      # [B, 256]
        out = self.fc(x)             # [B, num_classes]
        
        return out,x
    