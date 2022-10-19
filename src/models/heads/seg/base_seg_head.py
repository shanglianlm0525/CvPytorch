# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 18:39
# @Author : liumin
# @File : base_seg_head.py

from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn


class BaseSegHead(nn.Module, metaclass=ABCMeta):
    def __init__(self, num_classes, in_channels=None, channels=None, dropout_ratio=0.1, conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseSegHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.cls_seg = nn.Conv2d(self.channels, num_classes, kernel_size=1)

    def classify(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.cls_seg(feat)
        return output

    @abstractmethod
    def forward(self, x):
        pass
