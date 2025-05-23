import torch
import torch.nn as nn
import torch.nn.functional as F


class HumanVisionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, roi_kernel_size=3):
        """
        模仿人类视觉系统的小目标捕捉能力，设计动态注意力机制。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        :param roi_kernel_size: 用于生成ROI注意力的卷积核大小
        """
        super(HumanVisionAttention, self).__init__()

        # 通道缩减层，减少计算复杂度
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)

        # ROI（感兴趣区域）估计，通过卷积捕捉小目标区域
        self.roi_conv = nn.Conv2d(in_channels // reduction, 1, kernel_size=roi_kernel_size, stride=1,
                                  padding=roi_kernel_size // 2, bias=False)

        # 通道扩展层，恢复到原始通道数
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        # 权重生成层
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播：动态调整感兴趣区域的分辨率和处理优先级。
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 动态注意力调整后的特征图
        """
        # 1. 通道缩减，减少计算复杂度
        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        # 2. ROI估计，得到可能包含小目标的区域
        roi_map = self.sigmoid(self.roi_conv(reduced_features))  # 生成ROI的概率图，(B, 1, H, W)

        # 3. 高分辨率区域保持细节，低分辨率区域降采样以降低计算量
        high_res_area = reduced_features * roi_map  # 重点区域高分辨率处理
        low_res_area = F.interpolate(reduced_features * (1 - roi_map), scale_factor=0.5, mode='bilinear',
                                     align_corners=False)  # 非重点区域降分辨率

        # 4. 恢复通道数
        high_res_area = self.channel_expand(high_res_area)  # 恢复通道维度 (B, C, H, W)
        low_res_area = F.interpolate(self.channel_expand(low_res_area), size=x.shape[2:], mode='bilinear',
                                     align_corners=False)  # 恢复分辨率

        # 5. 动态调整权重，将高分辨率和低分辨率的区域结合
        final_output = high_res_area + low_res_area  # 汇总高低分辨率特征
        return final_output