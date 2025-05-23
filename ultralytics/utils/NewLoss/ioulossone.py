# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model validation metrics."""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

def bbox_shape_iou(box1, box2, xywh=True, scale=0.7, eps=1e-7):
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
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    '''
    Shape-IoU
    '''
    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance    #Shape-Distance  
    ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps                            # convex diagonal squared
    center_distance_x = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4
    center_distance_y = ((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    center_distance = hh * center_distance_x + ww * center_distance_y
    distance = center_distance / c2

    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    #Shape-Shape    
    omiga_w = hh * torch.abs(w1 - w2) / torch.max(w1, w2)
    omiga_h = ww * torch.abs(h1 - h2) / torch.max(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
    
    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU    #Shape-IoU
    iou = iou - distance - 0.5 * ( shape_cost)
    print('Shape-IoU🚀')
    return iou  # IoU


def bbox_mpdiou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MPDIoU=False, eps=1e-7):
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
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or MPDIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or MPDIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif MPDIoU:
                d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
                d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
                mpdiou_hw_pow = 10 ** 2 + 10 ** 2 # 关于这个高度和宽度
                # TODO
                print('MPDIoU🚀')
                return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU🚀
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    return iou  # IoU



def bbox_inner_multi_iou(box1, box2, ratio = 0.8, xywh=True, eps=1e-7, Inner_GIoU=False, Inner_DIoU=False, Inner_CIoU=False, Inner_EIoU=False, Inner_SIoU=False, Inner_WIoU=False, FocalLoss_=False): 
    (x1, y1, w1, h1) = box1.chunk(4, -1)
    (x2, y2, w2, h2) = box2.chunk(4, -1)
    # @from MangoAI &3836712GKcH2717GhcK.
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    inner_b1_x1, inner_b1_x2, inner_b1_y1, inner_b1_y2 = x1 - w1_*ratio, x1 + w1_*ratio,\
                                                             y1 - h1_*ratio, y1 + h1_*ratio
    inner_b2_x1,inner_b2_x2, inner_b2_y1, inner_b2_y2 = x2 - w2_*ratio, x2 + w2_*ratio,\
                                                             y2 - h2_*ratio, y2 + h2_*ratio
    inner_inter = (torch.min(inner_b1_x2, inner_b2_x2) - torch.max(inner_b1_x1, inner_b2_x1)).clamp(0) * \
                   (torch.min(inner_b1_y2, inner_b2_y2) - torch.max(inner_b1_y1, inner_b2_y1)).clamp(0)
    inner_union = w1*ratio * h1*ratio + w2*ratio * h2*ratio - inner_inter + eps
    inner_iou = inner_inter/inner_union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
    
    # =======SIoU
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
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
    # =======SIoU
    
    print('Inner-IoU🚀')
    inner_siou = inner_iou - 0.5 * (distance_cost + shape_cost)

    return inner_siou  # IoU


def bbox_piou(box1, box2, xywh=True, PIoU=False, PIoUv2=False, Lambda=1.3,eps=1e-7):

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

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union   
    
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height

    # PIoU
    dw1=torch.abs(b1_x2.minimum(b1_x1)-b2_x2.minimum(b2_x1))
    dw2=torch.abs(b1_x2.maximum(b1_x1)-b2_x2.maximum(b2_x1))
    dh1=torch.abs(b1_y2.minimum(b1_y1)-b2_y2.minimum(b2_y1))
    dh2=torch.abs(b1_y2.maximum(b1_y1)-b2_y2.maximum(b2_y1))
    P=((dw1+dw2)/torch.abs(w2)+(dh1+dh2)/torch.abs(h2))/4
    L_v1=1-iou - torch.exp(-P**2)+1
    
    
    if PIoU:
        print('PIoU🚀')        
        return L_v1 

    if PIoUv2:
        print('PIoUv2🚀')    
        q=torch.exp(-P)
        x=q*Lambda
        return  3*x*torch.exp(-x**2)*L_v1


def nwd(a, b):
    # @from MangoAI &3836712GKcH2717GhcK.
    b1_x1, b1_y1, b1_x2, b1_y2 = a.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b.chunk(4, -1)
    BX_L2Norm = torch.pow((b1_x1 - b2_x1), 2)
    BY_L2Norm = torch.pow((b1_y1 - b2_y1), 2)
    p1 = BX_L2Norm + BY_L2Norm
    w_FroNorm = torch.pow((b1_x2 - b2_x2)/2, 2)
    h_FroNorm = torch.pow((b1_y2 - b2_y2)/2, 2)
    p2 = w_FroNorm + h_FroNorm
    wasserstein = torch.exp(-torch.pow((p1+p2), 1 / 2) / 2.5)
    print('NWD🚀')
    return wasserstein