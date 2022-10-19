# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/18 20:50
# @Author : liumin
# @File : segformer_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(BaseSegHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, strides, **kwargs):
        super(SegFormerHead, self).__init__(**kwargs)
        assert len(strides) == len(self.in_channels)
        self.strides = strides

        self.linear_c4 = MLP(input_dim=self.in_channels[3], embed_dim=self.channels)
        self.linear_c3 = MLP(input_dim=self.in_channels[2], embed_dim=self.channels)
        self.linear_c2 = MLP(input_dim=self.in_channels[1], embed_dim=self.channels)
        self.linear_c1 = MLP(input_dim=self.in_channels[0], embed_dim=self.channels)

        self.linear_fuse = ConvModule(
            in_channels=self.channels*4,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        return self.classify(_c)