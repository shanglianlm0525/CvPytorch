# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 14:26
# @Author : liumin
# @File : fpn.py

"""
    Feature Pyramid Networks for Object Detection
    https://arxiv.org/abs/1612.03144
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.convs import ConvModule


class FPN(nn.Module):
    '''modified from MMDetection'''

    def __init__(self, in_channels,out_channels, add_extra_levels=False, extra_levels=2):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.add_extra_levels = add_extra_levels
        self.extra_levels = extra_levels

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i],out_channels, kernel_size=1, stride=1,
                padding=0, dilation=1, groups=1, bias=False,norm='BatchNorm2d',activation='ReLU')
            fpn_conv = ConvModule(out_channels,out_channels,kernel_size=3, stride=1,
                                  padding=1, dilation=1, groups=1, bias=False,norm='BatchNorm2d',activation='ReLU')
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        if self.add_extra_levels and self.extra_levels>0:
            for i in range(extra_levels):
                in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels, out_channels, kernel_size=3, stride=2,
                                    padding=1, dilation=1, groups=1, bias=False,norm='BatchNorm2d',activation='ReLU')
                self.fpn_convs.append(extra_fpn_conv)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        assert len(x) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.add_extra_levels and self.extra_levels>0:
            if self.add_extra_levels:
                # add conv layers on top of original feature maps (RetinaNet)
                for i in range(used_backbone_levels, used_backbone_levels+self.extra_levels):
                    outs.append(self.fpn_convs[i](outs[-1]))
            else:
                # use max pool to get more levels on top of outputs
                # (e.g., Faster R-CNN, Mask R-CNN)
                for i in range(self.extra_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))

        return tuple(outs)


if __name__ == '__main__':
    import torch
    in_channels = [2, 3, 5, 7]
    scales = [340, 170, 84, 43]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = FPN(in_channels, 11, True).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')