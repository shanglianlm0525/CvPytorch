# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/21 18:01
# @Author : liumin
# @File : topformer_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class TopFormerHead(BaseSegHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, is_depthwise=False, **kwargs):
        super(TopFormerHead, self).__init__(**kwargs)
        assert self.in_channels == self.channels
        self.linear_fuse = ConvModule(
            in_channels=self.channels, out_channels=self.channels, kernel_size=1, stride=1,
            groups=self.channels if is_depthwise else 1, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg
        )

    def forward(self, x):
        feat = x[0]
        for pred in x[1:]:
            feat += F.interpolate(pred, size=feat.size()[2:], mode='bilinear', align_corners=False)
        feat = self.linear_fuse(feat)
        feat = self.classify(feat)
        return feat



