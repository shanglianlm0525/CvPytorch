# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/10/26 10:31
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from src.models.anchors.efficientdet_anchor import EfficientDetAnchors
from src.models.anchors.faceboxes_priorbox import FaceBoxesPriorBox


__all__ = ['EfficientDetAnchors', 'FaceBoxesPriorBox']


def build_anchor(cfg):
    anchor_cfg = deepcopy(cfg)
    name = anchor_cfg.pop('name')
    if name == 'EfficientDetAnchors':
        return EfficientDetAnchors(**anchor_cfg)
    elif name == 'FaceBoxesPriorBox':
        return FaceBoxesPriorBox(**anchor_cfg)
    else:
        raise NotImplementedError
