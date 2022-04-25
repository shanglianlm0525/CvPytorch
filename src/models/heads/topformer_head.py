# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/21 18:01
# @Author : liumin
# @File : topformer_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.convs import ConvModule


class TopFormerHead(nn.Module):
    def __init__(self, num_classes=80, in_channels=[128, 128, 128], norm_cfg=dict(type='BN', requires_grad=True), activation='ReLU6'):
        super(TopFormerHead, self).__init__()
        self.num_classes = num_classes

        self.linear_fuse = ConvModule(in_channels, in_channels, 1, 1, 0, norm_cfg=norm_cfg, activation=activation)
        self.classifier = nn.Conv2d(in_channels, self.num_classes, 1, 1, 0)

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
        agg = x[0]
        for xx in x[1:]:
            agg += F.interpolate(xx, size=agg.size()[2:], mode='bilinear', align_corners=True)
        out = self.classifier(self.linear_fuse(agg))
        return out
