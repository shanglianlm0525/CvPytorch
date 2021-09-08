# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/23 14:13
# @Author : liumin
# @File : yolofastestv2_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.convs import ConvModule
from ..modules.init_weights import xavier_init


class DWConvblock(nn.Module):
    def __init__(self, input_channels, output_channels, size = 5):
        super(DWConvblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, size, 1, size//2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),

            nn.Conv2d(output_channels, output_channels, size, 1, size//2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            )

    def forward(self, x):
        x = self.block(x)
        return x


class YoloFastestv2Neck(nn.Module):
    def __init__(self, in_channels,out_channels, add_extra_levels=False, extra_levels=2, conv_cfg=None, norm_cfg=dict(type='BN'),activation='ReLU'):
        super(YoloFastestv2Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.add_extra_levels = add_extra_levels
        self.extra_levels = extra_levels

        self.lateral_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i], out_channels, kernel_size=1, stride=1,
                                padding=0, bias=False, norm_cfg=norm_cfg, activation=activation)
            cls_conv = DWConvblock(out_channels, out_channels)
            reg_conv = DWConvblock(out_channels, out_channels)
            self.lateral_convs.append(l_conv)
            self.cls_convs.append(cls_conv)
            self.reg_convs.append(reg_conv)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, x):
        assert len(x) == len(self.in_channels)
        x = list(x)
        x[0] = torch.cat((x[0], F.interpolate(x[1], scale_factor=2)), 1)

        reg_outs, cls_outs = [], []
        for i, (lateral_conv, reg_conv, cls_conv) in enumerate(zip(self.lateral_convs, self.reg_convs, self.cls_convs)):
            lateral = lateral_conv(x[i])
            reg_outs.append(reg_conv(lateral))
            cls_outs.append(cls_conv(lateral))

        return reg_outs, cls_outs, cls_outs # reg, obj, cls

