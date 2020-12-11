# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 14:26
# @Author : liumin
# @File : fpn.py

"""
    Feature Pyramid Networks for Object Detection
    https://arxiv.org/abs/1612.03144
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.convs import ConvModule


class FPN(nn.Module):
    '''modified from MMDetection'''

    def __init__(self, in_channels,out_channels):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i],out_channels, kernel_size=3, stride=1,
                padding=0, dilation=1, groups=1, bias=False)
            self.lateral_convs.append(l_conv)

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert len(x) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear')

        # build outputs
        outs = [
            # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
            laterals[i] for i in range(used_backbone_levels)
        ]
        return tuple(outs)