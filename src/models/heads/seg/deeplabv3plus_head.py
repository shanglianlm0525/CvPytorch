# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/17 10:34
# @Author : liumin
# @File : deeplabv3plus_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bricks import ConvModule, DepthwiseSeparableConvModule
from src.models.heads.seg.deeplabv3_head import ASPP, Deeplabv3Head


class DepthwiseSeparableASPPModule(ASPP):
    """Atrous Spatial Pyramid Pooling (ASPP) Module with depthwise separable
    conv."""

    def __init__(self, **kwargs):
        super(DepthwiseSeparableASPPModule, self).__init__(**kwargs)
        for i, dilation in enumerate(self.dilations):
            if dilation > 1:
                self[i] = DepthwiseSeparableConvModule(
                    self.in_channels,
                    self.channels,
                    3,
                    dilation=dilation,
                    padding=dilation,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)



class Deeplabv3PlusHead(Deeplabv3Head):
    def __init__(self, low_in_channels, low_channels, **kwargs):
        super(Deeplabv3PlusHead, self).__init__(**kwargs)

        self.aspp = DepthwiseSeparableASPPModule(
            dilations=self.dilations, in_channels=self.in_channels, channels=self.channels,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        if low_in_channels > 0:
            self.low_proj = ConvModule(
                low_in_channels, low_channels, 1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        else:
            self.low_proj = None

        self.fuse = nn.Sequential(
            DepthwiseSeparableConvModule(
                self.channels + low_channels, self.channels, 3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg),
            DepthwiseSeparableConvModule(
                self.channels, self.channels, 3, padding=1,
                norm_cfg=self.norm_cfg, act_cfg=self.act_cfg))

    def forward(self, x):
        outs = [F.interpolate(self.proj(x[1]), size=x[1].size()[2:], mode='bilinear', align_corners=False)]
        outs.extend(self.aspp(x[1]))
        outs = torch.cat(outs, dim=1)
        outs = self.reduce(outs)

        if self.low_proj is not None:
            low_outs = self.low_proj(x[0])
            outs = F.interpolate(outs, size=low_outs.size()[2:], mode='bilinear', align_corners=False)
            outs = torch.cat([outs, low_outs], dim=1)

        outs = self.fuse(outs)
        return self.classify(outs)


if __name__=='__main__':
    model = Deeplabv3PlusHead(num_classes=19, in_channels=2048, channels=512,
        dilations=(1, 12, 24, 36), low_in_channels=256, low_channels=48)
    print(model)