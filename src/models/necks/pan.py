# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 14:53
# @Author : liumin
# @File : pan.py

"""
    Path Aggregation Network for Instance Segmentation
    https://arxiv.org/abs/1803.01534
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .fpn import FPN

class PAN(FPN):
    '''modified from MMDetection'''

    def __init__(self, in_channels, out_channels, add_extra_levels=False, extra_levels=2):
        super().__init__(in_channels,out_channels, add_extra_levels, extra_levels)
        self.init_weights()

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
        # part 1: from original levels
        inter_outs = [
            laterals[i] for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            prev_shape = inter_outs[i + 1].shape[2:]
            inter_outs[i + 1] += F.interpolate(inter_outs[i], size=prev_shape, mode='bilinear')

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            inter_outs[i] for i in range(1, used_backbone_levels)
        ])

        return tuple(outs)