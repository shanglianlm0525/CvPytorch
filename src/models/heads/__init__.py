# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:59
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from .fcos_head import FcosHead
from .nanodet_head import NanodetHead

__all__ = ['FcosHead','NanodetHead']

def build_head(cfg):
    head_cfg = deepcopy(cfg)
    name = head_cfg.pop('name')
    if name == 'FcosHead':
        return FcosHead(**head_cfg)
    elif name == 'NanodetHead':
        return NanodetHead(**head_cfg)
    else:
        raise NotImplementedError
