# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:00
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .bifpn import BiFPN
from .fastestdet_neck import FastestDetNeck
from .fcos_fpn import FCOSFPN
from .fpn import FPN
from .ghost_pan import GhostPAN
from .giraffeneck import GiraffeNeck
from .lfd_neck import LFDNeck
from .rep_pan import RepPAN
from .pan import PAN
from .tan import TAN
from .yolop_neck import YOLOPNeck
from .yolov3_neck import YOLOv3Neck
from .yolov5_neck import YOLOv5Neck
from .yolov7_neck import YOLOv7Neck
from .yolox_neck import YOLOXNeck
from .pai_yolox_neck import PAI_YOLOXNeck

__all__ = [
    'FPN',
    'PAN',
    'TAN',
    'BiFPN',
    'YOLOXNeck',
    'PAI_YOLOXNeck',
    'YOLOv3Neck',
    'YOLOv5Neck',
    'YOLOv7Neck',
    'YOLOPNeck',
    'GhostPAN',
    'FCOSFPN',
    'LFDNeck',
    'RepPAN',
    'FastestDetNeck',
    'GiraffeNeck'
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
    elif name == 'YOLOXNeck':
        return YOLOXNeck(**neck_cfg)
    elif name == 'PAI_YOLOXNeck':
        return PAI_YOLOXNeck(**neck_cfg)
    elif name == 'YOLOv3Neck':
        return YOLOv3Neck(**neck_cfg)
    elif name == 'YOLOv5Neck':
        return YOLOv5Neck(**neck_cfg)
    elif name == 'YOLOv7Neck':
        return YOLOv7Neck(**neck_cfg)
    elif name == 'YOLOPNeck':
        return YOLOPNeck(**neck_cfg)
    elif name == 'GhostPAN':
        return GhostPAN(**neck_cfg)
    elif name == 'FCOSFPN':
        return FCOSFPN(**neck_cfg)
    elif name == 'RepPAN':
        return RepPAN(**neck_cfg)
    elif name == 'FastestDetNeck':
        return FastestDetNeck(**neck_cfg)
    elif name == 'GiraffeNeck':
        return GiraffeNeck(**neck_cfg)
    elif name == 'LFDNeck':
        return LFDNeck(**neck_cfg)
    else:
        raise NotImplementedError
