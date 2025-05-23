# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "AdaptiveResolutionAttention",
    "HumanVisionAttention",
    "PyramidAdaptiveResolutionAttention",
    "ImprovedAdaptiveResolutionAttention"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)



import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义自适应分辨率注意力模块
class AdaptiveResolutionAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        自适应分辨率注意力模块，用于在目标大小不同的情况下动态调整分辨率。
        :param in_channels: 输入通道数
        :param reduction: 通道缩减系数，用于控制注意力的计算复杂度
        """
        super(AdaptiveResolutionAttention, self).__init__()
        # 通道缩减与扩展层
        self.channel_reduce = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, stride=1)
        self.channel_expand = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, stride=1)

        # 分辨率调整层（上采样、下采样）
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
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

        # 1. 通道缩减
        reduced_features = F.relu(self.channel_reduce(x))  # (B, C/reduction, H, W)

        # 2. 动态调整分辨率
        upsampled = self.upsample(reduced_features)  # 上采样特征
        downsampled = self.downsample(reduced_features)  # 下采样特征

        # 3. 恢复通道维度
        upsampled_expanded = F.relu(self.channel_expand(upsampled))  # (B, C, 2H, 2W)
        downsampled_expanded = F.relu(self.channel_expand(downsampled))  # (B, C, H/2, W/2)

        # 4. 使用双线性插值将所有特征调整到原始大小
        upsampled_resized = F.interpolate(upsampled_expanded, size=original_size, mode='bilinear', align_corners=True)
        downsampled_resized = F.interpolate(downsampled_expanded, size=original_size, mode='bilinear', align_corners=True)

        # 5. 权重生成
        combined_features = upsampled_resized + downsampled_resized  # 将上采样和下采样特征结合
        attention_weights = self.sigmoid(combined_features)  # 生成权重

        # 6. 进行权重分配
        x_weighted = x * attention_weights
        return x_weighted



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

        # 跨尺度卷积融合
        self.cross_scale_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1,
                                          groups=in_channels)

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
