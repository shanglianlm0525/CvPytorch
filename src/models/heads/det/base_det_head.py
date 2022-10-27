# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 14:31
# @Author : liumin
# @File : base_det_head.py

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn


class BaseDetHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes=None, in_channels=None, channels=None, strides=[8, 16, 32], depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseDetHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        self.strides = strides
        self.depthwise = depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

    @abstractmethod
    def forward(self, x):
        pass