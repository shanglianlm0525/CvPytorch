# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:00
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .bifpn import BiFPN
from .fpn import FPN
from .ghost_pan import GhostPAN
from .tph_yolov5_neck import TPH_YOLOv5Neck
from .yolofastestv2_neck import YoloFastestv2Neck
from .pan import PAN
from .tan import TAN
from .yolop_neck import YOLOPNeck
from .yolov3_neck import YOLOv3Neck
from .yolov5_neck import YOLOv5Neck
from .yolox_fpn import YOLOXNeck

__all__ = [
    'FPN',
    'PAN',
    'TAN',
    'BiFPN',
    'YoloFastestv2Neck',
    'YOLOXNeck',
    'YOLOv3Neck',
    'YOLOv5Neck',
    'TPH_YOLOv5Neck',
    'YOLOPNeck',
    'GhostPAN'
]

def build_neck(cfg):
    neck_cfg = deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'FPN':
        return FPN(**neck_cfg)
    elif name == 'PAN':
        return PAN(**neck_cfg)
    elif name == 'TAN':
        return TAN(**neck_cfg)
    elif name == 'BiFPN':
        return BiFPN(**neck_cfg)
    elif name == 'YoloFastestv2Neck':
        return YoloFastestv2Neck(**neck_cfg)
    elif name == 'YOLOXNeck':
        return YOLOXNeck(**neck_cfg)
    elif name == 'YOLOv3Neck':
        return YOLOv3Neck(**neck_cfg)
    elif name == 'YOLOv5Neck':
        return YOLOv5Neck(**neck_cfg)
    elif name == 'TPH_YOLOv5Neck':
        return TPH_YOLOv5Neck(**neck_cfg)
    elif name == 'YOLOPNeck':
        return YOLOPNeck(**neck_cfg)
    elif name == 'GhostPAN':
        return GhostPAN(**neck_cfg)
    else:
        raise NotImplementedError
