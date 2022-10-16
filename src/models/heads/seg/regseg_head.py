# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/7 12:03
# @Author : liumin
# @File : regseg_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_seg_head import BaseSegHead
from ...bricks import ConvModule


class RegSegHead(BaseSegHead):
    def __init__(self, mid_channels = [8, 128], **kwargs):
        super(RegSegHead, self).__init__(**kwargs)

        self.head4 = ConvModule(self.in_channels[0], mid_channels[0], kernel_size=1,
                   norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.head8 = ConvModule(self.in_channels[1], mid_channels[1], kernel_size=1,
                   norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.head16 = ConvModule(self.in_channels[2], mid_channels[1], kernel_size=1,
                   norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.conv8 = ConvModule(mid_channels[1], self.channels, kernel_size=3, stride=1, padding=1,
                   norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.conv4 = ConvModule(self.channels + mid_channels[0], self.channels, kernel_size=3, stride=1, padding=1,
                   norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        x4, x8, x16 = x
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)
        x4 = self.conv4(x4)
        return self.classify(x4)