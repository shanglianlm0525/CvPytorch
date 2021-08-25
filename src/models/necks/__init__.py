# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:00
# @Author : liumin
# @File : __init__bak.py

from copy import deepcopy
from .fpn import FPN
from .light_fpn import LightFPN
from .pan import PAN
from .tan import TAN
from .yolov3_neck import YOLOv3Neck
from .yolov5_neck import YOLOv5Neck
from .yolox_fpn import YOLOXNeck

__all__ = [
    'FPN',
    'PAN',
    'TAN',
    'LightFPN',
    'YOLOXNeck',
    'YOLOv3Neck',
    'YOLOv5Neck'
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
    elif name == 'LightFPN':
        return LightFPN(**neck_cfg)
    elif name == 'YOLOXNeck':
        return YOLOXNeck(**neck_cfg)
    elif name == 'YOLOv3Neck':
        return YOLOv3Neck(**neck_cfg)
    elif name == 'YOLOv5Neck':
        return YOLOv5Neck(**neck_cfg)
    else:
        raise NotImplementedError
