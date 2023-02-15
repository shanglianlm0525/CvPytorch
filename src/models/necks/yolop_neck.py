# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/7 13:26
# @Author : liumin
# @File : yolop_neck.py

"""
    Path Aggregation Network for Instance Segmentation
    https://arxiv.org/abs/1803.01534
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.yolo11_modules import C3, Conv


class YOLOPNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOPNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.up_convs = nn.ModuleList()
        for i in range(self.num_ins - 1, 0, -1):
            if i == self.num_ins - 1:
                up_conv = nn.Sequential(
                    BottleneckCSP(self.in_channels[i], self.in_channels[i], 1, False),
                    Conv(self.in_channels[i], self.in_channels[i - 1], 1, 1, 0)
                )
                self.up_convs.append(up_conv)
            else:
                up_conv = nn.Sequential(
                    BottleneckCSP(self.in_channels[i] * 2, self.in_channels[i], 1, False),
                    Conv(self.in_channels[i], self.in_channels[i - 1], 1, 1, 0)
                )
                self.up_convs.append(up_conv)

        self.down_convs1 = nn.ModuleList()
        self.down_convs2 = nn.ModuleList()
        for i in range(self.num_ins):
            if i == 0:
                self.down_convs1.append(BottleneckCSP(self.in_channels[i] * 2, self.out_channels[i], 3, False))
            else:
                self.down_convs1.append(BottleneckCSP(self.in_channels[i], self.out_channels[i], 3, False))
            if i < self.num_ins - 1:
                self.down_convs2.append(Conv(self.out_channels[i], self.in_channels[i], 3, 2, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        outs = x
        used_backbone_levels = len(outs)
        for i in range(used_backbone_levels - 1, 0, -1):
            outs[i] = self.up_convs[used_backbone_levels - 1 - i](outs[i])
            lateral_up = F.interpolate(outs[i], size=x[i - 1].shape[2:], mode='nearest')
            outs[i - 1] = torch.cat([lateral_up, outs[i - 1]], dim=1)

        seg_outs = outs.copy()
        for i in range(used_backbone_levels):
            if i < used_backbone_levels - 1:
                outs[i] = self.down_convs1[i](outs[i])
                outs[i + 1] = torch.cat([self.down_convs2[i](outs[i]), outs[i + 1]], dim=1)
            else:
                outs[i] = self.down_convs1[i](outs[i])

        return seg_outs, outs


if __name__ == "__main__":
    import torch

    in_channels = [128, 256, 512]
    out_channels = [128, 256, 512]
    scales = [80, 40, 20]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = YOLOPNeck(in_channels, out_channels).eval()
    print(self)
    outputs1, outputs2 = self.forward(inputs)
    for i in range(len(outputs1)):
        print(f'outputs[{i}].shape = {outputs1[i].shape}')
    for i in range(len(outputs2)):
        print(f'outputs[{i}].shape = {outputs2[i].shape}')
