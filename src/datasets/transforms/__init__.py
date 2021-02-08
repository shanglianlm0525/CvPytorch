# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/5 14:09
# @Author : liumin
# @File : __init__.py

import copy
from CvPytorch.src.datasets.transforms import custom_transforms as ctf

def build_transforms(cfg, mode='train'):
    _params = []
    transform_cfg = copy.deepcopy(cfg)
    for t, v in transform_cfg.items():
        t = getattr(ctf, t)(**v) if v is not None else getattr(ctf, t)()
        _params.append(t)
    return ctf.Compose(_params)