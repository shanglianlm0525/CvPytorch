# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/1/7 8:45
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from src.models.detects.fcos_detect import FCOSDetect


__all__ = [
    'FCOSDetect'
]


def build_detect(cfg):
    detect_cfg = deepcopy(cfg)
    name = detect_cfg.pop('name')

    if name == 'FCOSDetect':
        return FCOSDetect(**detect_cfg)
    else:
        raise NotImplementedError