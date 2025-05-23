import torch
import numpy as np

'''
YOLOv8 + gwdloss 损失函数" 改进
- 只需要加上对应改进的核心 gwdloss 损失函数模块，该项目代码就可以直接运行各种`YOLOv8-xxx.yaml`网络配置文件，乐高式创新改进，一键运行即可
使用 gwdloss 损失函数 进行实验改进
- 项目相关改进可以支持 答疑 服务。详情见 ⭐⭐⭐ https://github.com/iscyy/ultralyticsPro/wiki/YOLOv8 ⭐⭐⭐ 说明
'''

#


import torch
import torch.nn as nn
import numpy as np

# 定义 GWD + NWD 损失函数
def gwdloss(pbox, gtbox, fg_mask, alpha=1.0, tau=2.0, reduction='mean'):
    """
    Generalized Wasserstein Distance (GWD) 损失函数 + NWD改进版
    :param pbox: 预测框 (B, 4) 格式为 (x_center, y_center, w, h)
    :param gtbox: 实际框 (B, 4) 格式为 (x_center, y_center, w, h)
    :param fg_mask: 前景掩码，表明哪些框为正样本 (B, 1)
    :param alpha: 控制损失平滑性的参数
    :param tau: 归一化参数
    :param reduction: 损失的归一化方式，可以是 'mean' 或 'sum'
    :return: GWD + NWD 损失值
    """
    # 提取中心点和宽高
    p_x, p_y, p_w, p_h = pbox[:, 0], pbox[:, 1], pbox[:, 2], pbox[:, 3]
    gt_x, gt_y, gt_w, gt_h = gtbox[:, 0], gtbox[:, 1], gtbox[:, 2], gtbox[:, 3]

    # 计算中心点的 L2 距离 (Wasserstein Distance)
    center_distance = (p_x - gt_x)**2 + (p_y - gt_y)**2

    # 计算宽高之间的 Wasserstein 距离
    wh_distance = (p_w - gt_w)**2 + (p_h - gt_h)**2

    # 计算标准化后的 GWD 损失（带有尺度调整因子）
    gwd_loss = (center_distance + wh_distance) / (2 * tau**2)

    # 引入 NWD 归一化，防止尺度变化过大
    nwd_loss = torch.log(1 + gwd_loss / alpha)

    # 应用前景掩码
    nwd_loss = nwd_loss * fg_mask
    print('GWDloss🚀')
    # 损失函数的归一化
    if reduction == 'mean':
        return torch.mean(nwd_loss)
    elif reduction == 'sum':
        return torch.sum(nwd_loss)
    else:
        return nwd_loss

# 示例使用
# pbox 和 gtbox 是 YOLOv8 输出的预测框和实际框
# fg_mask 是前景掩码，指示哪些框是前景
pbox = torch.tensor([[50, 50, 100, 150], [30, 40, 90, 120]], dtype=torch.float32)
gtbox = torch.tensor([[45, 55, 95, 145], [35, 45, 85, 115]], dtype=torch.float32)
fg_mask = torch.tensor([1, 1], dtype=torch.float32)  # 所有框都是前景框

# 计算损失
loss = gwdloss(pbox, gtbox, fg_mask)
print("GWD + NWD Loss:", loss.item())

