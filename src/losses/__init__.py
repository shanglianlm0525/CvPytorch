# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/2 13:21
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from src.losses.fcos_loss import FCOSLoss
from src.losses.ssd_loss import SsdLoss
from src.losses.yolofastestv2_loss import YoloFastestv2Loss
from src.losses.yolop_loss import YolopLoss
from src.losses.yolov5_loss import Yolov5Loss
from src.losses.efficientdet_loss import EfficientDetLoss
from src.losses.yolox_loss import YoloxLoss


__all__ = ['Yolov5Loss', 'EfficientDetLoss', 'YoloFastestv2Loss', 'YoloxLoss', 'SsdLoss',
           'YolopLoss', 'FCOSLoss']



def build_loss(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'Yolov5Loss':
        return Yolov5Loss(**loss_cfg)
    elif name == 'EfficientDetLoss':
        return EfficientDetLoss(**loss_cfg)
    elif name == 'YoloFastestv2Loss':
        return YoloFastestv2Loss(**loss_cfg)
    elif name == 'YoloxLoss':
        return YoloxLoss(**loss_cfg)
    elif name == 'SsdLoss':
        return SsdLoss(**loss_cfg)
    elif name == 'FCOSLoss':
        return FCOSLoss(**loss_cfg)

    elif name == 'YolopLoss':
        return YolopLoss(**loss_cfg)
    else:
        raise NotImplementedError