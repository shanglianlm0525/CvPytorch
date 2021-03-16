# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/5 14:09
# @Author : liumin
# @File : __init__.py

import copy
import CvPytorch.src.transforms.seg_transforms as seg_t
import CvPytorch.src.transforms.ins_transforms as ins_t

def build_transforms(dict_name, cfg, mode='train'):
    _params = []

    if dict_name == 'CLS_CLASSES':
        trans = seg_t
    elif dict_name == 'SEG_CLASSES':
        trans = seg_t
    elif dict_name == 'DET_CLASSES':
        trans = seg_t
    elif dict_name == 'INS_CLASSES':
        trans = ins_t
    else:
        raise ValueError("Unsupported transforms type: {}".format(dict_name))

    transform_cfg = copy.deepcopy(cfg)
    for t, v in transform_cfg.items():
        t = getattr(trans, t)(**v) if v is not None else getattr(trans, t)()
        _params.append(t)
    return trans.Compose(_params)