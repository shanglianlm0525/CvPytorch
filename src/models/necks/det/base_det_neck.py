# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 13:24
# @Author : liumin
# @File : base_det_neck.py

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn


class BaseDetNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self, in_channels=None, out_channels=None, aux_out_channels=None, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseDetNeck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_out_channels = aux_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

    @abstractmethod
    def forward(self, x):
        pass