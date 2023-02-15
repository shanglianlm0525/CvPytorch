# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/2/7 14:36
# @Author : liumin
# @File : base_det_detect.py

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn


class BaseDetDetect(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes=None, in_channels=None, channels=None, strides=[8, 16, 32], depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseDetDetect, self).__init__()
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