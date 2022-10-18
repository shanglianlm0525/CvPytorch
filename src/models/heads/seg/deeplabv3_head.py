# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/17 10:47
# @Author : liumin
# @File : aspp_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class ASPP(nn.ModuleList):
    """Atrous Spatial Pyramid Pooling (ASPP) Module.

    Args:
        dilations (tuple[int]): Dilation rate of each layer.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, dilations, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg):
        super(ASPP, self).__init__()
        self.dilations = dilations
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for dilation in dilations:
            self.append(ConvModule(self.in_channels, self.channels, 1 if dilation == 1 else 3,
                    dilation=dilation, padding=0 if dilation == 1 else dilation,
                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, x):
        """Forward function."""
        outs = []
        for aspp_module in self:
            outs.append(aspp_module(x))
        return outs


class Deeplabv3Head(BaseSegHead):
    """
    Args:
        dilations (tuple[int]): Dilation rates for ASPP module.
            Default: (1, 6, 12, 18).
    """
    def __init__(self, dilations=(1, 6, 12, 18), **kwargs):
        super(Deeplabv3Head, self).__init__(**kwargs)
        self.dilations = dilations

        self.proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(self.in_channels, self.channels, 1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

        self.aspp = ASPP(dilations, self.in_channels, self.channels,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.reduce = ConvModule((len(dilations) + 1) * self.channels,self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        outs = [F.interpolate(self.proj(x), size=x.size()[2:], mode='bilinear', align_corners=False)]
        outs.extend(self.aspp(x))
        outs = torch.cat(outs, dim=1)
        outs = self.reduce(outs)
        return self.classify(outs)