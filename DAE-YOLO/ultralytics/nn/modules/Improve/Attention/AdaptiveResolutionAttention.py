import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):

        super(AdaptiveResolutionAttention, self).__init__()

        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1, bias=False)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1, bias=False)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        original_size = x.shape[2:]

        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        downsampled = self.downsample(reduced_features)  # (B, C/reduction, H/2, W/2)

        downsampled_expanded = F.relu(self.channel_expand(downsampled))  # (B, C, H/2, W/2)
        downsampled_resized = F.interpolate(downsampled_expanded, size=original_size, mode='bilinear', align_corners=False)

        attention_weights = self.sigmoid(downsampled_resized)  # 生成权重

        x_weighted = x * attention_weights
        return x_weighted


