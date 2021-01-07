# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 14:06
# @Author : liumin
# @File : activations.py

import torch
import torch.nn as nn

__avalible__ = {
    'ReLU': nn.ReLU,
   'LeakyReLU': nn.LeakyReLU,
   'ReLU6': nn.ReLU6,
   'PReLU': nn.PReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'Hardswish': nn.Hardswish,
    'Hardshrink': nn.Hardshrink
}


def build_activate_layer(name, **params):
    assert name in __avalible__.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(**params, inplace=False)
    else:
        return __avalible__[name](inplace=False)



