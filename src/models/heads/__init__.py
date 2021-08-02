# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:59
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .deeplabv3_head import Deeplabv3Head
from .deeplabv3plus_head import Deeplabv3PlusHead
from .nanodet_head import NanoDetHead
from .stdc_head import StdcHead
from .yolov5_head import YOLOv5Head

__all__ = [
    'YOLOv5Head',
    'YOLOXHead',
    'NanoDetHead',
    'Deeplabv3Head',
    'Deeplabv3PlusHead',
    'StdcHead'
]

from .yolox_head import YOLOXHead


def build_head(cfg):
    head_cfg = deepcopy(cfg)
    name = head_cfg.pop('name')

    if name == 'YOLOv5Head':
        return YOLOv5Head(**head_cfg)
    elif name == 'YOLOXHead':
        return YOLOXHead(**head_cfg)
    elif name == 'NanoDetHead':
        return NanoDetHead( **head_cfg)

    elif name == 'Deeplabv3Head':
        return Deeplabv3Head(**head_cfg)
    elif name == 'Deeplabv3PlusHead':
        return Deeplabv3PlusHead(**head_cfg)
    elif name == 'StdcHead':
        return StdcHead(**head_cfg)
    else:
        raise NotImplementedError