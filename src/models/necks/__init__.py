# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:00
# @Author : liumin
# @File : __init__bak.py

from copy import deepcopy
from .fpn import FPN
from .pan import PAN
from .yolo_neck import YOLOV3Neck, YOLOV4Neck, YOLOV5Neck

__all__ = [
    'FPN', 'PAN',
    'YOLOV3Neck', 'YOLOV4Neck', 'YOLOV5Neck'
]

def build_neck(cfg):
    neck_cfg = deepcopy(cfg)
    name = neck_cfg.pop('name')
    if name == 'FPN':
        return FPN(**neck_cfg)
    elif name == 'PAN':
        return PAN(**neck_cfg)
    else:
        raise NotImplementedError
