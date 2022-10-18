# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/17 19:58
# @Author : liumin
# @File : psp_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, **kwargs):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1,
                        conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg,**kwargs)))

    def forward(self, x):
        """Forward function."""
        x_size = x.size()[2:]
        ppm_outs = []
        for f in self:
            ppm_out = F.interpolate(f(x), size=x_size, mode='bilinear', align_corners=False)
            ppm_outs.append(ppm_out)
        return ppm_outs


class PSPHead(BaseSegHead):
    """Pyramid Scene Parsing Network.
    Args:
        bins (tuple[int]): Pooling scales used in Pooling Pyramid
            Module. Default: (1, 2, 3, 6).
    """
    def __init__(self, bins=(1, 2, 3, 6), **kwargs):
        super(PSPHead, self).__init__(**kwargs)
        self.bins = bins
        self.psp = PPM(
            self.bins, self.in_channels, self.channels,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.bottleneck = ConvModule(
            self.in_channels + len(self.bins) * self.channels, self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        output = [x]
        output.extend(self.psp(x))
        output = torch.cat(output, dim=1)
        output = self.bottleneck(output)
        return self.classify(output)