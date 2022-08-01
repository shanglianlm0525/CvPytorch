# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/1 14:11
# @Author : liumin
# @File : lspnet_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSPNetHead(nn.Module):
    def __init__(self, num_classes, channels):
        super(LSPNetHead, self).__init__()
        self.num_classes = num_classes
        self.channels = channels

        self.classifier = nn.Conv2d(self.channels[-1] * 2, self.num_classes, 1, 1, 0, bias=True)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xh, xl = x
        x_cat = torch.cat([xh, F.interpolate(xl, (int(xh.shape[-2]), int(xh.shape[-1])), mode='bilinear', align_corners=True)], dim=1)
        return self.classifier(x_cat)