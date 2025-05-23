import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidAdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, num_scales=3):
        """
        改进后的自适应分辨率注意力模块，结合了特征金字塔机制，适合小目标检测。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        :param num_scales: 需要生成的金字塔层数
        """
        super(PyramidAdaptiveResolutionAttention, self).__init__()

        self.num_scales = num_scales

        # 通道缩减与扩展层
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        # 下采样层，使用金字塔多尺度结构
        self.downsamples = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_scales)])

        # 跨尺度的权重生成
        self.sigmoid = nn.Sigmoid()

        # 通道融合层
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        """
        前向传播过程中，生成金字塔结构，并通过融合金字塔不同尺度的特征来调整权重。
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 自适应调整后的特征图
        """
        # 原始特征图大小
        original_size = x.shape[2:]

        # 1. 通道缩减，减少计算复杂度
        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        # 2. 生成多尺度金字塔
        pyramid_features = [reduced_features]
        for downsample in self.downsamples:
            pyramid_features.append(downsample(pyramid_features[-1]))  # 逐级下采样

        # 3. 上采样并恢复每个尺度特征至原始大小，同时进行融合
        upsampled_features = []
        for features in pyramid_features:
            upsampled_features.append(F.interpolate(features, size=original_size, mode='bilinear', align_corners=False))

        # 4. 融合所有尺度的特征
        fused_features = torch.stack(upsampled_features, dim=0).sum(dim=0)  # 对所有上采样后的特征进行求和融合

        # 5. 恢复通道维度
        fused_features = F.relu(self.channel_expand(fused_features))  # (B, C, H, W)

        # 6. 生成权重
        attention_weights = self.sigmoid(fused_features)  # 生成权重

        # 7. 将权重应用到原始输入
        x_weighted = x * attention_weights

        # 8. 融合原始输入和加权后的输入
        output = x + x_weighted  # 残差连接，保留原始信息

        return output
