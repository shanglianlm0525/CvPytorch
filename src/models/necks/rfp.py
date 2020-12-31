# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:02
# @Author : liumin
# @File : rfp.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import build_backbone
from ..modules.init_weights import kaiming_init
from ..necks import FPN

"""
    DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution
    https://arxiv.org/pdf/2006.02334.pdf
"""

class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out

class RFP(FPN):
    def __init__(self, rfp_steps, rfp_backbone, aspp_out_channels, aspp_dilations=(1, 3, 6, 1), **kwargs):
        super(RFP, self).__init__(**kwargs)
        self.rfp_steps = rfp_steps
        self.rfp_modules = nn.ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

    def forward(self, x):
        x = list(x)
        assert len(x) == len(self.in_channels) + 1  # +1 for input image
        img = x.pop(0)
        # FPN forward
        out = super().forward(tuple(x))
        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [out[0]] + list(
                self.rfp_aspp(out[i]) for i in range(1, len(out)))
            out_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            # FPN forward
            out_idx = super().forward(out_idx)
            out_new = []
            for ft_idx in range(len(out_idx)):
                add_weight = torch.sigmoid(self.rfp_weight(out_idx[ft_idx]))
                out_new.append(add_weight * out_idx[ft_idx] +
                             (1 - add_weight) * out[ft_idx])
            out = out_new
        return out