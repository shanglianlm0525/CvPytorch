# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 18:18
# @Author : liumin
# @File : fcn_head.py

import torch
import torch.nn as nn

from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class FCNHead(BaseSegHead):
    """
        Fully Convolutional Networks for Semantic Segmentation
        https://arxiv.org/pdf/1411.4038.pdf
    """
    def __init__(self, num_convs=2, kernel_size=3, is_concat=True, dilation=1, **kwargs):
        super(FCNHead, self).__init__(**kwargs)
        self.num_convs = num_convs
        self.is_concat = is_concat
        self.kernel_size = kernel_size

        if num_convs == 0:
            self.convs = nn.Identity()
            assert self.in_channels == self.channels
        else:
            conv_padding = (kernel_size // 2) * dilation
            convs = []
            convs.append(
                ConvModule(self.in_channels,self.channels,kernel_size=kernel_size,padding=conv_padding,dilation=dilation,
                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            for i in range(num_convs - 1):
                convs.append(
                    ConvModule(self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation,
                        conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))
            self.convs = nn.Sequential(*convs)

        if self.is_concat:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels, self.channels,
                kernel_size=kernel_size, padding=kernel_size // 2,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

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
        feats = self.convs(x)
        if self.is_concat:
            feats = self.conv_cat(torch.cat([x, feats], dim=1))
        output = self.classify(feats)
        return output


if __name__ == '__main__':
    model = FCNHead(num_classes=19, in_channels=256, channels=256, num_convs=1,dropout_ratio=0.0, is_concat = False)
    print(model)