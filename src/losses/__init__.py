# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/2 13:21
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from src.losses.fastestdet_loss import FastestDetLoss
from src.losses.fcos_loss import FCOSLoss
from src.losses.objectbox_loss import ObjectBoxLoss
from src.losses.seg import CrossEntropyLoss2d
from src.losses.seg import OhemCrossEntropyLoss2d
from src.losses.seg import DetailAggregateLoss
from src.losses.yolop_loss import YolopLoss
from src.losses.yolov5_loss import YOLOv5Loss
from src.losses.efficientdet_loss import EfficientDetLoss
from src.losses.yolov7_loss import YOLOv7Loss



# Image Classification


# Semantic Segmentation


# Object Detectiton
from .det import YOLOXLoss
from .det import YOLOv6Loss

__all__ = ['YOLOv5Loss', 'YOLOXLoss', 'YOLOv7Loss','EfficientDetLoss',
           'YolopLoss', 'FCOSLoss', 'YOLOv6Loss', 'FastestDetLoss', 'ObjectBoxLoss']



def build_loss(cfg):
    loss_cfg = deepcopy(cfg)
    name = loss_cfg.pop('name')

    if name == 'YOLOv5Loss':
        return YOLOv5Loss(**loss_cfg)
    elif name == 'EfficientDetLoss':
        return EfficientDetLoss(**loss_cfg)
    elif name == 'YOLOXLoss':
        return YOLOXLoss(**loss_cfg)
    elif name == 'YOLOv6Loss':
        return YOLOv6Loss(**loss_cfg)
    elif name == 'YOLOv7Loss':
        return YOLOv7Loss(**loss_cfg)
    elif name == 'FCOSLoss':
        return FCOSLoss(**loss_cfg)
    elif name == 'FastestDetLoss':
        return FastestDetLoss(**loss_cfg)
    elif name == 'ObjectBoxLoss':
        return ObjectBoxLoss(**loss_cfg)

    # seg
    elif name == 'CrossEntropyLoss2d':
        return CrossEntropyLoss2d(**loss_cfg)
    elif name == 'OhemCrossEntropyLoss2d':
        return OhemCrossEntropyLoss2d(**loss_cfg)
    elif name == 'DetailAggregateLoss':
        return DetailAggregateLoss(**loss_cfg)
    elif name == 'YolopLoss':
        return YolopLoss(**loss_cfg)
    else:
        raise NotImplementedError(name)