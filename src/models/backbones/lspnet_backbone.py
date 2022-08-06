# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/1 13:23
# @Author : liumin
# @File : lspnet_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.convs import ConvModule


class LSPNetBackbone(nn.Module):
    def __init__(self, subtype='', resolutions=[0.75, 0.25], depths=[1, 3, 3, 10, 10], channels=[8, 24, 48, 96, 96]):
        super(LSPNetBackbone, self).__init__()
        self.subtype = subtype
        self.resolutions = resolutions
        self.depths = depths
        self.channels = channels
        self.high_net = BaseNet(self.channels, self.depths)
        self.low_net = BaseNet(self.channels, self.depths)

    def _preprocess_input(self, x):
        r1 = self.resolutions[0]
        r2 = self.resolutions[1]
        x1 = F.interpolate(x, (int(x.shape[-2] * r1), int(x.shape[-1] * r1)), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x, (int(x.shape[-2] * r2), int(x.shape[-1] * r2)), mode='bilinear', align_corners=True)
        return x1, x2

    def forward(self, x):
        xh, xl = self._preprocess_input(x)
        xh, xl = self.high_net.stages[0](xh), self.low_net.stages[0](xl)
        xh, xl = self.high_net.stages[1](xh), self.low_net.stages[1](xl)
        xh, xl = self.high_net.stages[2](xh), self.low_net.stages[2](xl)
        xh, xl = bi_interaction(xh, xl)
        xh, xl = self.high_net.stages[3](xh), self.low_net.stages[3](xl)
        xh, xl = bi_interaction(xh, xl)
        xh, xl = self.high_net.stages[4](xh), self.low_net.stages[4](xl)
        return xh, xl


def upsample(x, size):
    if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
        return F.interpolate(x, size, mode='bilinear', align_corners=True)
    else:
        return x


def bi_interaction(x_h, x_l):
    sizeH = (int(x_h.shape[-2]), int(x_h.shape[-1]))
    sizeL = (int(x_l.shape[-2]), int(x_l.shape[-1]))
    o_h = x_h + upsample(x_l, sizeH)
    o_l = x_l + upsample(x_h, sizeL)
    return o_h, o_l


def tr_interaction(x1, x2, x3):
    s1 = (int(x1.shape[-2]), int(x1.shape[-1]))
    s2 = (int(x2.shape[-2]), int(x2.shape[-1]))
    s3 = (int(x3.shape[-2]), int(x3.shape[-1]))
    o1 = x1 + upsample(x2, s1) + upsample(x3, s1)
    o2 = x2 + upsample(x1, s2) + upsample(x3, s2)
    o3 = x3 + upsample(x2, s3) + upsample(x1, s3)
    return o1, o2, o3


class BaseNet(nn.Module):
    def __init__(self, channels=[8, 24, 48, 96, 96], depths=[1, 3, 3, 10, 10], strides=[2, 2, 2, 2, 1]):
        super(BaseNet, self).__init__()
        self.depths = depths
        self.channels = channels
        self.strides = strides

        self.stages = nn.ModuleList()
        c_in = 3
        for l, c, s in zip(self.depths, self.channels, self.strides):
            self.stages.append(self._make_stage(c_in, c, l, s))
            c_in = c

    @staticmethod
    def _make_stage(c_in, c_out, depth, stride):
        layers = []
        for i in range(depth):
            layers.append(ConvModule(c_in if i == 0 else c_out, c_out, 3, stride if i == 0 else 1, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for s in self.stages:
            out = s(out)
        return out