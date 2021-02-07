# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/5 14:09
# @Author : liumin
# @File : __init__.py

import copy

['KeepRatio',
'RandomHorizontalFlip',
'RandomTranslation',
'ToTensor',
'Normalize']

def build_transforms(cfg, mode=None):
    _params = []
    transform_cfg = copy.deepcopy(cfg)
    for t, v in transform_cfg.items():
        print("'" + t + "', ")