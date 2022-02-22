# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/2/20 18:33
# @Author : liumin
# @File : regseg_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class RegSegHead(nn.Module):
    def __init__(self, num_classes, in_channels = [48, 128, 320]):
        super(RegSegHead, self).__init__()

        self.head4 = ConvBnAct(in_channels[0], 8, 1)
        self.head8 = ConvBnAct(in_channels[1], 128, 1)
        self.head16 = ConvBnAct(in_channels[2], 128, 1)

        self.conv8 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv4 = ConvBnAct(64+8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
        x4 = self.classifier(x4)
        return x4