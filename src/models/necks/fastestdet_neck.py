# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/4 19:26
# @Author : liumin
# @File : fastestdet_neck.py

import torch
import torch.nn as nn


class SPP(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SPP, self).__init__()
        self.Conv1x1 = nn.Sequential(
                            nn.Conv2d(input_channels, output_channels, 1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )

        self.S1 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            )

        self.S2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            )

        self.S3 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_channels, output_channels, 5, 1, 2, groups=output_channels, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
            )

        self.output = nn.Sequential(nn.Conv2d(output_channels * 3, output_channels, 1, 1, 0, bias=False),
                                    nn.BatchNorm2d(output_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Conv1x1(x)

        y1 = self.S1(x)
        y2 = self.S2(x)
        y3 = self.S3(x)

        y = torch.cat((y1, y2, y3), dim=1)
        y = self.relu(x + self.output(y))
        return y


class FastestDetNeck(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], out_channels=96):
        super(FastestDetNeck, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.SPP = SPP(sum(in_channels), out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x

        P5 = self.upsample(C5)
        P3 = self.avg_pool(C3)
        P_cat = torch.cat((P3, C4, P5), dim=1)

        return self.SPP(P_cat)