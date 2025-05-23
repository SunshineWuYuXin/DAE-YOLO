import torch
import torch.nn as nn
import torch.nn.functional as F

# 改进后的自适应分辨率注意力模块
class AdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的自适应分辨率注意力模块，减少不必要的计算，提升效率。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        """
        super(AdaptiveResolutionAttention, self).__init__()
        # 通道缩减与扩展层
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        # 下采样层
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 权重生成层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程中，根据输入特征图动态调整分辨率。
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 自适应调整后的特征图
        """
        # 原始特征图大小
        original_size = x.shape[2:]

        # 1. 通道缩减，减少计算复杂度
        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        # 2. 下采样特征
        downsampled = self.downsample(reduced_features)  # (B, C/reduction, H/2, W/2)

        # 3. 恢复通道维度并上采样到原始尺寸
        downsampled_expanded = F.relu(self.channel_expand(downsampled))  # (B, C, H/2, W/2)
        downsampled_resized = F.interpolate(downsampled_expanded, size=original_size, mode='bilinear', align_corners=False)

        # 4. 权重生成
        attention_weights = self.sigmoid(downsampled_resized)  # 生成权重

        # 5. 将权重应用到原始输入
        x_weighted = x * attention_weights
        return x_weighted


