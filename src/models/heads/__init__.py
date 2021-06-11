# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:59
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .nanodet_head import NanoDetHead
from .yolov5_head import YOLOv5Head

__all__ = [
    'YOLOv5Head',
    'NanoDetHead'
]

def build_head(cfg):
    head_cfg = deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'YOLOv5Head':
        return YOLOv5Head(**head_cfg)
    elif name == 'NanoDetHead':
        return NanoDetHead(**head_cfg)
    else:
        raise NotImplementedError