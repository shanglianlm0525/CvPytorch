# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/23 14:13
# @Author : liumin
# @File : light_fpn.py

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


class LightFPN(nn.Module):
    def __init__(self, in_channels,out_channels, add_extra_levels=False, extra_levels=2, conv_cfg=None, norm_cfg=None,activation=None):
        super(LightFPN, self).__init__()
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
                                padding=0, dilation=1, groups=1, bias=False, norm_cfg=norm_cfg, activation=activation)
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
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False)

        # build outputs
        reg_outs = [
            self.reg_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        cls_outs = [
            self.cls_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        return reg_outs, cls_outs, cls_outs # reg, obj, cls


if __name__ == '__main__':
    import torch
    in_channels = [2, 3, 5]
    scales = [340, 170, 84]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = LightFPN(in_channels, 11, False).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')