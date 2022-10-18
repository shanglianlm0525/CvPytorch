# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/18 14:46
# @Author : liumin
# @File : uper_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead
from src.models.heads.seg.psp_head import PPM


class UPerHead(BaseSegHead):
    """Unified Perceptual Parsing for Scene Understanding.
    Args:
        bins (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, bins=(1, 2, 3, 6), **kwargs):
        super(UPerHead, self).__init__(**kwargs)

        # PSP Module
        self.ppm = PPM(bins, self.in_channels[-1], self.channels,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.ppm_bottleneck = ConvModule(
            self.in_channels[-1] + len(bins) * self.channels,
            self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels, self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


    def forward(self, x):
        """Forward function."""

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Forward function of PSP module.
        psp_outs = [x[-1]]
        psp_outs.extend(self.ppm(x[-1]))
        psp_outs = torch.cat(psp_outs, dim=1)
        psp_outs = self.ppm_bottleneck(psp_outs)
        laterals.append(psp_outs)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_outs = self.fpn_bottleneck(fpn_outs)
        return self.classify(fpn_outs)

