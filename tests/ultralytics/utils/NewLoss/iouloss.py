# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def bbox_multi_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, EfficiCIoU=False, XIoU=False, is_Focaler='None', eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # @from MangoAI &3836712GKcH2717GhcK.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    '''
        当使用 ultralytics\cfg\models\cfg2024\YOLOv8-Loss\Inner 文件下的内容时

        1. 需要将这串代码的注释 取消掉

        2. 将本文件95.96行代码注释掉
    
    '''
    # ====Inner相关代码===========================================================================================

    # ratio = 0.8
    # (x1, y1, w1, h1) = box1.chunk(4, -1)
    # (x2, y2, w2, h2) = box2.chunk(4, -1)
    # # @from MangoAI &3836712GKcH2717GhcK.
    # w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    # b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    # b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    # inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
    #                                                          y1 - h1_*ratio, y1 + h1_*ratio
    # inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
    #                                                          y2 - h2_*ratio, y2 + h2_*ratio
    # inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
    #                (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    # inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
    # inner_iou = inner_inter/inner_union
    # iou = inner_iou
    # print('使用Inner系列')

    # cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    # ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    
    # ===============================================================================================

    # Get the coordinates of bounding boxes

    Inner = True
    if Inner == False:
        if xywh:  # transform from xywh to xyxy
            (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
            w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
            b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
            b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        else:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
            b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
        ).clamp_(0)

        # Union Area
        union = w1 * h1 + w2 * h2 - inter + eps

        # IoU
        iou = inter / union
    #  --------------原始IoU----------------------
    else:
        (x1, y1, w1, h1) = box1.chunk(4, -1)
        (x2, y2, w2, h2) = box2.chunk(4, -1)
        # @from MangoAI &3836712GKcH2717GhcK.
        ratio = 0.8
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
        inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
                                                                y1 - h1_*ratio, y1 + h1_*ratio
        inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
                                                                y2 - h2_*ratio, y2 + h2_*ratio
        inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                    (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
        union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
        print('use Inner系列🍈')
        iou = inner_inter/union
    #  --------------Inner----------------------


    # FocalerIoU 改进
    Focaler = False
    if Focaler:
        d=0.0
        u=0.95
        print('use Focaler系列🍈')
        iou = ((iou - d) / (u - d)).clamp(0, 1)

    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU or EfficiCIoU or XIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or EfficiCIoU or XIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('CIoU🚀')
                return iou - (rho2 / c2 + v * alpha)  # CIoU🚀
            elif SIoU:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                print('SIoU🚀')
                return iou - 0.5 * (distance_cost + shape_cost)# SIoU🚀
            elif EIoU:
                v = torch.pow(1 / (1 + torch.exp(-(w2 / h2))) - 1 / (1 + torch.exp(-(w1 / h1))), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('EIoU🚀')
                return iou - (rho2 / c2 + v * alpha)# EIoU🚀
            elif EfficiCIoU:
                # @from MangoAI &3836712GKcH2717GhcK.
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                        (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                w_dis=torch.pow(b1_x2-b1_x1-b2_x2+b2_x1, 2)
                h_dis=torch.pow(b1_y2-b1_y1-b2_y2+b2_y1, 2)
                cw2=torch.pow(cw , 2)+eps
                ch2=torch.pow(ch , 2)+eps
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                print('EfficiCIoU🚀')
                return iou - (rho2 / c2 + w_dis/cw2+h_dis/ch2 + v * alpha)
            elif XIoU:# @from MangoAI &3836712GKcH2717GhcK.
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
                beta = 1
                q2 = (1 + torch.exp(-(w2 / h2)))
                q1 = (1 + torch.exp(-(w1 / h1)))
                v = torch.pow(1 / q2 - 1 / q1, 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps)) * beta
                print('XIoU🚀')
                return iou - (rho2 / c2 + v * alpha)
            print('DIoU🚀')
            return iou - rho2 / c2  # DIoU🚀
        elif WIoU:
            from ultralytics.utils.NewLoss.wiou import IoU_Cal
            b1 = torch.stack([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
            b2 = torch.stack([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)
            '''
                monotonous: {
                None: origin
                True: monotonic FM
                False: non-monotonic FM
            }
            '''
            self = IoU_Cal(b1, b2, monotonous = False)  # monotonous set WIoUv1、WIoUv2、WIoUv3
            loss = getattr(IoU_Cal, 'WIoU')(b1, b2, self=self)
            iou = 1 - self.iou
            print('WIoU🚀')
            return loss, iou# WIoU🚀
        c_area = cw * ch + eps  # convex area
        print('GIoU🚀')
        return iou - (c_area - union) / c_area  # GIoU🚀 https://arxiv.org/pdf/1902.09630.pdf
    return iou  # 🚀IoU


def bbox_focal_multi_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, WIoU=False, FocalLoss_= 'none', eps=1e-7):
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): A tensor representing a single bounding box with shape (1, 4).
        box2 (torch.Tensor): A tensor representing n bounding boxes with shape (n, 4).
        xywh (bool, optional): If True, input boxes are in (x, y, w, h) format. If False, input boxes are in
                               (x1, y1, x2, y2) format. Defaults to True.
        GIoU (bool, optional): If True, calculate Generalized IoU. Defaults to False.
        DIoU (bool, optional): If True, calculate Distance IoU. Defaults to False.
        CIoU (bool, optional): If True, calculate Complete IoU. Defaults to False.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.
        # @from MangoAI &3836712GKcH2717GhcK.

    Returns:
        (torch.Tensor): IoU, GIoU, DIoU, or CIoU values depending on the specified flags.
    """

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                if FocalLoss_ == 'Focal_CIoU':
                    print(' use focal ciou 🍇')
                    return iou - (rho2 / c2 + v * alpha), (inter/(union + eps)) ** 0.5# mg
                print('CIoU🚀')
                return iou - (rho2 / c2 + v * alpha)  # CIoU🚀
            elif SIoU:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                if FocalLoss_ == 'Focal_SIoU':
                    print(' use focal-siou 🍇')
                    return iou - 0.5 * (distance_cost + shape_cost), (inter/(union + eps)) ** 0.5
                print('SIoU🚀')
                return iou - 0.5 * (distance_cost + shape_cost)# SIoU🚀
            elif EIoU:
                w_dis=torch.pow(b1_x2-b1_x1-b2_x2+b2_x1, 2)
                h_dis=torch.pow(b1_y2-b1_y1-b2_y2+b2_y1, 2)
                cw2=torch.pow(cw , 2)+eps
                ch2=torch.pow(ch , 2)+eps
                if FocalLoss_ == 'Focal_EIoU':
                    print(' use focal-eiou 🍇')
                    return iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2), (inter/(union + eps)) ** 0.5
                print('EIoU🚀')
                return iou - (rho2 / c2 + w_dis / cw2 + h_dis / ch2)# EIoU🚀
            if FocalLoss_ == 'Focal_DIoU':
                    print(' use focal-diou 🍇')
                    return iou - rho2 / c2, (inter/(union + eps)) ** 0.5
            print('DIoU🚀')
            return iou - rho2 / c2  # DIoU🚀
        elif WIoU:
            from ultralytics.utils.NewLoss.wiou import IoU_Cal
            b1 = torch.stack([b1_x1, b1_y1, b1_x2, b1_y2], dim=-1)
            b2 = torch.stack([b2_x1, b2_y1, b2_x2, b2_y2], dim=-1)
            '''
                monotonous: {
                None: origin
                True: monotonic FM
                False: non-monotonic FM
            }
            '''
            self = IoU_Cal(b1, b2, monotonous = False)  # monotonous set WIoUv1、WIoUv2、WIoUv3
            loss = getattr(IoU_Cal, 'WIoU')(b1, b2, self=self)
            iou = 1 - self.iou
            if FocalLoss_ == 'Focal_WIoU':
                print(' use focal-wiou 🍇')
                return iou, (inter/(union + eps)) ** 0.5, loss
            print('WIoU🚀')
            return loss, iou# WIoU🚀
        c_area = cw * ch + eps  # convex area
        if FocalLoss_ == 'Focal_GIoU':
                print(' use focal-giou 🍇')
                return iou - (c_area - union) / c_area, (inter/(union + eps)) ** 0.5
        print('GIoU🚀')
        return iou - (c_area - union) / c_area  # GIoU🚀 https://arxiv.org/pdf/1902.09630.pdf
    return iou  # 🚀IoU

