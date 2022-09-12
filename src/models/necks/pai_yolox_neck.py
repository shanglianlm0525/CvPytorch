# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/6 15:31
# @Author : liumin
# @File : pai_yolox_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.necks import YOLOXNeck
from src.models.necks.asff import ASFF


class PAI_YOLOXNeck(YOLOXNeck):
    def __init__(self, subtype='pai_yolox_s', channels=[256, 512, 1024], asff_channel=2, expand_kernel=3, depth_mul=1.0, width_mul=1.0, act='SiLU'):
        super(PAI_YOLOXNeck, self).__init__(subtype=subtype, channels=channels, depth_mul=depth_mul, width_mul=width_mul)
        assert isinstance(channels, list)

        self.asff_1 = ASFF(level=0,asff_channel=asff_channel,expand_kernel=expand_kernel,multiplier=width_mul,act=act)
        self.asff_2 = ASFF(level=1,asff_channel=asff_channel,expand_kernel=expand_kernel,multiplier=width_mul,act=act)
        self.asff_3 = ASFF(level=2,asff_channel=asff_channel,expand_kernel=expand_kernel,multiplier=width_mul,act=act)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        outs = super(PAI_YOLOXNeck, self).forward(x)

        asff_out0 = self.asff_1(outs)
        asff_out1 = self.asff_2(outs)
        asff_out2 = self.asff_3(outs)
        outputs = (asff_out2, asff_out1, asff_out0)
        return outputs