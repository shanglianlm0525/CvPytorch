# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/18 16:20
# @Author : liumin
# @File : deeplabv3_head.py

from torch import nn
from src.models.modules.aspp import ASPP


class Deeplabv3Head(nn.Module):
    def __init__(self, in_channels, dilations, num_classes):
        super(Deeplabv3Head, self).__init__()
        mid_channels = 256
        self.aspp = ASPP(inplanes=in_channels, dilations = dilations)

        self.classifier = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(mid_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x):
        x = self.aspp(x)
        return self.classifier(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)