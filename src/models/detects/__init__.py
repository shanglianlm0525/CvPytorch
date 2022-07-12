# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/7 8:45
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from src.models.detects.fcos_detect import FCOSDetect
from src.models.detects.yolov5_detect import Yolov5Detect
from src.models.detects.yolov6_detect import Yolov6Detect
from src.models.detects.yolov7_detect import Yolov7Detect

__all__ = [
    'FCOSDetect',
    'Yolov5Detect',
    'Yolov6Detect',
    'Yolov7Detect'
]


def build_detect(cfg):
    detect_cfg = deepcopy(cfg)
    name = detect_cfg.pop('name')

    if name == 'FCOSDetect':
        return FCOSDetect(**detect_cfg)
    elif name == 'Yolov5Detect':
        return Yolov5Detect(**detect_cfg)
    elif name == 'Yolov6Detect':
        return Yolov6Detect(**detect_cfg)
    elif name == 'Yolov7Detect':
        return Yolov7Detect(**detect_cfg)
    else:
        raise NotImplementedError