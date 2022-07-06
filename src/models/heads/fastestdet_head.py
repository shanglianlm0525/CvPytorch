# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/4 19:36
# @Author : liumin
# @File : fastestdet_head.py

import torch
import torch.nn as nn


class FastestDetHead(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(FastestDetHead, self).__init__()
        self.num_classes = num_classes

        self.conv1x1 = nn.Sequential(
                            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True)
                        )

        self.obj_layers = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups = in_channels, bias = False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, 1, 1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(1)
                            )

        self.reg_layers = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups = in_channels, bias = False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, 4, 1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(4)
                            )
        self.cls_layers = nn.Sequential(
                                nn.Conv2d(in_channels, in_channels, 5, 1, 2, groups = in_channels, bias = False),
                                nn.BatchNorm2d(in_channels),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(in_channels, self.num_classes, 1, stride=1, padding=0, bias=False),
                                nn.BatchNorm2d(self.num_classes)
                            )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.conv1x1(x)

        obj = self.sigmoid(self.obj_layers(x))
        reg = self.reg_layers(x)
        cls = self.softmax(self.cls_layers(x))

        return torch.cat((obj, reg, cls), dim=1)
