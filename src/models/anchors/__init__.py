# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/10/26 10:31
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from src.models.anchors.efficientdet_anchor import EfficientDetAnchors


__all__ = ['EfficientDetAnchors']


def build_anchor(cfg):
    anchor_cfg = deepcopy(cfg)
    name = anchor_cfg.pop('name')
    if name == 'EfficientDetAnchors':
        return EfficientDetAnchors(**anchor_cfg)
    else:
        raise NotImplementedError
