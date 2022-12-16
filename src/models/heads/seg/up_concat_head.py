# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/12/14 17:22
# @Author : liumin
# @File : up_concat_head.py

import torch
import torch.nn as nn

from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class UpConcatHead(BaseSegHead):
    """
       IncepFormer: Efficient Inception Transformer with Pyramid Pooling for Semantic Segmentation
    """

    def __init__(self, **kwargs):
        super(UpConcatHead, self).__init__(**kwargs)
        self.linear_fuse = ConvModule(in_channels=sum(self.in_channels), out_channels=self.channels, kernel_size=1, norm_cfg=self.norm_cfg)


    def forward(self, x):
        feats = torch.cat(x, dim=1)
        feats = self.linear_fuse(feats)
        feats = self.classify(feats)
        return feats