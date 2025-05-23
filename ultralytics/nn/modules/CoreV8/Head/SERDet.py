# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Model head modules."""

import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from ultralytics.nn.modules.block import DFL, Proto
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import bias_init_with_prob

import torch.nn.functional as F

from ultralytics.nn.modules import DWConv
class CLS(nn.Module):
    def __init__(self, x, c3):
        super().__init__()
        self.cl_1 = DWConv(x, c3, 3)
        self.cl_2 = Conv(c3, c3, 3)
    def forward(self, x):
        q1 = self.cl_1(x)
        x1 = self.cl_2(q1)
        return x1


class SaELayer(nn.Module):
    def __init__(self, in_channel, reduction=32):
        super(SaELayer, self).__init__()
        assert in_channel>=reduction and in_channel%reduction==0,'invalid in_channel in SaElayer'
        self.reduction = reduction
        self.cardinality=4
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #cardinality 1
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel,in_channel//self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 2
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 3
        self.fc3 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        # cardinality 4
        self.fc4 = nn.Sequential(
            nn.Linear(in_channel, in_channel // self.reduction, bias=False),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_channel//self.reduction*self.cardinality, in_channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y1 = self.fc1(y)
        y2 = self.fc2(y)
        y3 = self.fc3(y)
        y4 = self.fc4(y)
        y_concate = torch.cat([y1,y2,y3,y4],dim=1)
        y_ex_dim = self.fc(y_concate).view(b,c,1,1)

        return x * y_ex_dim.expand_as(x)
        
class RE_(nn.Module):
    def __init__(self, x, c2, reg_max):
        super().__init__()
        self.cl_1 = Conv(x, c2, 3)
        self.cl_2 = Conv(c2, c2, 3)
        self.sev2 = SaELayer(c2, c2)
    def forward(self, x):
        q1 = self.cl_1(x)
        q1 = self.sev2(q1)
        x1 = self.cl_2(q1)
        return x1
        
class SERDet(nn.Module):
    """YOLOv8 SERDet head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.cv2_1 = nn.ModuleList(RE_(x, c2, reg_max = self.reg_max) for x in ch) 
        self.cv2_ = nn.ModuleList(nn.Conv2d(c2, 4 * self.reg_max, 1) for x in ch)

        self.cv3_1 = nn.ModuleList(CLS(x, c3) for x in ch) 
        self.cv3_ = nn.ModuleList(nn.Conv2d(c3, self.nc, 1) for x in ch)


    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2_[i](self.cv2_1[i](x[i])), self.cv3_[i](self.cv3_1[i](x[i]))), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xywh with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_h = shape[2] * self.stride[0]
            img_w = shape[3] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=dbox.device).reshape(1, 4, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)


    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2_, m.cv3_, m.stride):  # from
            a.bias.data[:] = 1.0  # box
            b.bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

