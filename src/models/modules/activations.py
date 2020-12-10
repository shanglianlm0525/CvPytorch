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
   'SELU': nn.SELU,
   'ELU': nn.ELU,
   None: nn.Identity
}


def activate_layers(name,**params):
    assert name in __avalible__.keys()
    if name == 'LeakyReLU':
        return nn.LeakyReLU(**params, inplace=True)
    else:
        return __avalible__[name](inplace=True)



