# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/3 10:45
# @Author : liumin
# @File : fcos_fpn.py

import torch.nn as nn
import torch.nn.functional as F
import math


class FCOSFPN(nn.Module):
    '''only for resnet50,101,152'''

    def __init__(self, in_channels=[512, 1024, 2048], out_channels=256):
        super(FCOSFPN, self).__init__()
        self.prj_3 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.prj_4 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.prj_5 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.conv_5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_out6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv_out7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        self.init_weights()

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]), mode='nearest')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P3 = self.prj_3(C3)
        P4 = self.prj_4(C4)
        P5 = self.prj_5(C5)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]