# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/2/7 13:34
# @Author : liumin
# @File : base_yolo_head.py

import math
from abc import abstractmethod, ABCMeta

import torch
import torch.nn as nn

from src.models.heads.det.base_det_head import BaseDetHead


class BaseYOLOHead(BaseDetHead):
    def __init__(self, subtype='yolov6_s', cfg=None, num_classes=80, in_channels=None, channels=None, out_channels=None, num_blocks=None, depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU')):
        super(BaseYOLOHead, self).__init__()
        self.subtype = subtype
        self.cfg = cfg
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.channels = channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.depthwise = depthwise
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        if self.channels is not None:
            self.channels = list(map(lambda x: max(round(x * width_mul), 1), self.channels))
        if self.out_channels is not None:
            self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), self.out_channels))
        if self.num_blocks is not None:
            self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    def forward(self, x):
        """Forward function."""
        pass