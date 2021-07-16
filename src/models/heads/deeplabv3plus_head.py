# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/18 16:22
# @Author : liumin
# @File : deeplabv3plus_head.py

import torch
from torch import nn
import torch.nn.functional as F
from src.models.modules.aspp import ASPP


class Deeplabv3PlusHead(nn.Module):
    def __init__(self, num_classes, low_level_channels, in_channels, dilations):
        super(Deeplabv3PlusHead, self).__init__()
        project_channels = 48
        mid_channels = 256
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, project_channels, 1, bias=False),
                                    nn.BatchNorm2d(project_channels),
                                    nn.ReLU(inplace=True))

        self.aspp = ASPP(inplanes=in_channels, dilations = dilations)

        self.classifier = nn.Sequential(nn.Conv2d(project_channels+mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(mid_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.project(low_level_feat)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        return self.classifier(torch.cat((x, low_level_feat), dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)