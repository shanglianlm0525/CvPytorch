# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:59
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .deeplabv3_head import Deeplabv3Head
from .deeplabv3plus_head import Deeplabv3PlusHead
from .nanodet_head import NanoDetHead
from .openpose_head import OpenPoseHead
from .stdc_head import StdcHead
from .yolo_fastestv2_head import YOLOFastestv2Head
from .yolop_head import YOLOPHead
from .yolov5_head import YOLOv5Head
from .yolox_head import YOLOXHead

__all__ = [
    'YOLOv5Head',
    'YOLOXHead',
    'NanoDetHead',
    'YOLOFastestv2Head',
    'YOLOPHead',
    'Deeplabv3Head',
    'Deeplabv3PlusHead',
    'StdcHead',
    'OpenPoseHead'
]


def build_head(cfg):
    head_cfg = deepcopy(cfg)
    name = head_cfg.pop('name')

    if name == 'YOLOv5Head':
        return YOLOv5Head(**head_cfg)
    elif name == 'YOLOXHead':
        return YOLOXHead(**head_cfg)
    elif name == 'NanoDetHead':
        return NanoDetHead( **head_cfg)
    elif name == 'YOLOFastestv2Head':
        return YOLOFastestv2Head( **head_cfg)
    elif name == 'YOLOPHead':
        return YOLOPHead( **head_cfg)

    elif name == 'Deeplabv3Head':
        return Deeplabv3Head(**head_cfg)
    elif name == 'Deeplabv3PlusHead':
        return Deeplabv3PlusHead(**head_cfg)
    elif name == 'StdcHead':
        return StdcHead(**head_cfg)

    elif name == 'OpenPoseHead':
        return OpenPoseHead(**head_cfg)
    else:
        raise NotImplementedError