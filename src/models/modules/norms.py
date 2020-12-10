# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:15
# @Author : liumin
# @File : norms.py

import torch
import torch.nn as nn


__avalible__ = {
    'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d,'BatchNorm3d': nn.BatchNorm3d
    ,'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d, 'InstanceNorm3d': nn.InstanceNorm3d
    ,'GroupNorm': nn.GroupNorm, 'SyncBatchNorm': nn.SyncBatchNorm, 'LayerNorm': nn.LayerNorm
}


def norm_layers(name,**params):
    assert name in __avalible__.keys()
    return __avalible__[name](**params)