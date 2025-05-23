import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        Squeeze-and-Excitation注意力模块，用于提升全局信息提取能力。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数
        """
        super(SEAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(avg_pool)
        x = F.relu(x)
        x = self.fc2(x)
        return x * self.sigmoid(x)

class ImprovedAdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        改进后的多尺度自适应分辨率注意力模块，结合SE注意力机制来提升全局信息的表达能力。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        """
        super(ImprovedAdaptiveResolutionAttention, self).__init__()

        # 通道缩减与扩展层
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        # 局部和全局多尺度特征融合
        self.local_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，用于全局上下文
        self.global_pool = nn.AdaptiveAvgPool2d(4)  # 小尺度全局池化，用于捕捉细节

        # 下采样层，保留原来的下采样机制
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        # 跨尺度卷积融合，确保输入通道数与 groups 参数匹配
        self.cross_scale_conv = nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1, stride=1, groups=in_channels // reduction)

        # 引入SE注意力机制，增强全局信息提取
        self.se_attention = SEAttention(in_channels)

        # 权重生成层，使用Softmax生成不同尺度的权重
        self.softmax = nn.Softmax(dim=1)

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

        # 2. 局部特征处理（下采样和多尺度特征融合）
        downsampled = self.downsample(reduced_features)  # (B, C/reduction, H/2, W/2)
        local_features = self.local_pool(downsampled)  # (B, C/reduction, 1, 1)
        global_features = self.global_pool(downsampled)  # (B, C/reduction, 4, 4)

        # 跨尺度卷积
        cross_scale_features = self.cross_scale_conv(downsampled)  # (B, C/reduction, H/2, W/2)

        # 3. 恢复通道维度并上采样到原始尺寸
        downsampled_expanded = F.relu(self.channel_expand(cross_scale_features))  # (B, C, H/2, W/2)
        downsampled_resized = F.interpolate(downsampled_expanded, size=original_size, mode='bilinear',
                                            align_corners=False)

        # 4. 融合全局和局部特征
        local_global_fused = downsampled_resized + F.interpolate(local_features + global_features, size=original_size,
                                                                 mode='bilinear', align_corners=False)

        # 5. 引入SE注意力机制
        se_fused = self.se_attention(local_global_fused)

        # 6. 使用Softmax生成不同尺度的自适应权重
        attention_weights = self.softmax(se_fused)

        # 7. 将权重应用到原始输入
        x_weighted = x * attention_weights
        return x_weighted

