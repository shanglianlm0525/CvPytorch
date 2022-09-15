# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/11 10:04
# @Author : liumin
# @File : yolov7_head.py

import torch
import torch.nn as nn
from src.models.modules.yolov7_modules import RepConv, Conv


class YOLOv7Head(nn.Module):
    def __init__(self, in_channels=[128, 256, 512], out_channels=[256, 512, 1024], depth_mul=1.0, width_mul=1.0):
        super(YOLOv7Head, self).__init__()
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))
        out_channels = list(map(lambda x: int(x * width_mul), out_channels))

        self.conv1 = RepConv(in_channels[0], out_channels[0])
        self.conv2 = RepConv(in_channels[1], out_channels[1])
        self.conv3 = RepConv(in_channels[2], out_channels[2])

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        return [x1, x2, x3]
