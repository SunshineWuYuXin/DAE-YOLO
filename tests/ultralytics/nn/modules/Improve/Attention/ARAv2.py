import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算平均值注意力
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算最大值注意力
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接两个注意力
        x = self.conv1(x)  # 通过卷积生成空间注意力
        return torch.sigmoid(x)


# 改进后的自适应分辨率注意力模块
class AdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的自适应分辨率注意力模块，包含通道注意力、空间注意力和多尺度融合机制。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        """
        super(AdaptiveResolutionAttention, self).__init__()
        # 通道缩减与扩展层
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)

        # 多尺度卷积用于特征融合
        self.fpn = nn.ModuleList([
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=1, bias=False),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1, bias=False)
        ])

        # 空间注意力机制
        self.spatial_attention = SpatialAttention()

        # 门控机制
        self.gate = nn.Linear(in_channels // reduction, in_channels // reduction)

        # 下采样层
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 权重生成层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播过程中，根据输入特征图动态调整分辨率和生成注意力权重。
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 自适应调整后的特征图
        """
        # 原始特征图大小
        original_size = x.shape[2:]

        # 1. 通道缩减，减少计算复杂度
        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        # 2. 下采样特征
        downsampled = self.downsample(reduced_features)  # (B, C/reduction, H/2, W/2)

        # 3. 多尺度特征融合
        scale1 = F.relu(self.fpn[0](downsampled))  # 第一层卷积
        scale2 = F.relu(self.fpn[1](downsampled))  # 第二层卷积（多尺度）
        combined_features = scale1 + scale2  # 多尺度特征融合

        # 4. 门控机制，动态控制信息流动
        gate_values = torch.mean(combined_features, dim=[2, 3])  # 全局池化得到通道信息
        gate_values = torch.sigmoid(self.gate(gate_values))  # 生成门控值
        gated_features = combined_features * gate_values.view(-1, combined_features.size(1), 1, 1)

        # 5. 恢复通道维度并上采样到原始尺寸
        expanded_features = F.relu(self.channel_expand(gated_features))  # 通道扩展
        resized_features = F.interpolate(expanded_features, size=original_size, mode='bilinear', align_corners=False)

        # 6. 生成通道注意力权重
        attention_weights = self.sigmoid(resized_features)  # 生成通道注意力权重

        # 7. 空间注意力机制
        spatial_attention_weights = self.spatial_attention(x)  # 生成空间注意力权重

        # 8. 融合通道和空间注意力
        x_weighted = x * attention_weights * spatial_attention_weights

        return x_weighted
