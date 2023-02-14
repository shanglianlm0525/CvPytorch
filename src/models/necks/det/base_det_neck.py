# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 13:24
# @Author : liumin
# @File : base_det_neck.py

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn


class BaseDetNeck(nn.Module, metaclass=ABCMeta):
    def __init__(self, subtype=None, cfg=None, in_channels=None, mid_channels=None, out_channels=None, num_blocks=None, aux_out_channels=None, depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseDetNeck, self).__init__()
        self.subtype = subtype
        self.cfg = cfg
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.aux_out_channels = aux_out_channels
        self.depthwise = depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        if self.mid_channels is not None:
            self.mid_channels = list(map(lambda x: max(round(x * width_mul), 1), self.mid_channels))
        if self.out_channels is not None:
            self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), self.out_channels))
        if self.num_blocks is not None:
            self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))

    @abstractmethod
    def forward(self, x):
        pass