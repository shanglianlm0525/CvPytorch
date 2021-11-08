# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/21 14:29
# @Author : liumin
# @File : bifpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.activations import Swish, MemoryEfficientSwish
from src.models.modules.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding, SeparableConvBlock

"""
    EfficientDet: Scalable and Efficient Object Detection
    https://arxiv.org/pdf/1911.09070.pdf
"""

class BiFPN(nn.Module):
    def __init__(self, in_channels, out_channels, first_time=False, attention=True, epsilon = 1e-4, conv_cfg=None, norm_cfg=None, activation=None):
        super(BiFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.first_time = first_time
        self.attention = attention
        self.epsilon = epsilon

        self.swish = MemoryEfficientSwish() # Swish()

        if self.first_time:
            # add extra conv layers
            self.p5_to_p6 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
                MaxPool2dStaticSamePadding(3, 2)
            )
            self.p6_to_p7 = MaxPool2dStaticSamePadding(3, 2)

            self.p5_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p4_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p3_down_channel = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[0], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

            self.p4_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[1], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )
            self.p5_down_channel_2 = nn.Sequential(
                Conv2dStaticSamePadding(in_channels[2], out_channels, 1),
                nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3),
            )

        # Conv layers
        self.conv6_up = SeparableConvBlock(out_channels)
        self.conv5_up = SeparableConvBlock(out_channels)
        self.conv4_up = SeparableConvBlock(out_channels)
        self.conv3_up = SeparableConvBlock(out_channels)

        self.conv4_down = SeparableConvBlock(out_channels)
        self.conv5_down = SeparableConvBlock(out_channels)
        self.conv6_down = SeparableConvBlock(out_channels)
        self.conv7_down = SeparableConvBlock(out_channels)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)

        # Weight
        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)


        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        illustration of a minimal bifpn unit
            P7_0 -------------------------> P7_2 -------->
               |-------------|                ↑
                             ↓                |
            P6_0 ---------> P6_1 ---------> P6_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P5_0 ---------> P5_1 ---------> P5_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |-------------|--------------↑ ↑
                             |--------------↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        if self.first_time:
            p3, p4, p5 = x
            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            # P3_0, P4_0, P5_0, P6_0 and P7_0
            p3_in, p4_in, p5_in, p6_in, p7_in = x

        # P7_0 to P7_2

        # Weights for P6_0 and P7_0 to P6_1
        p6_w1 = F.relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        p6_up = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * F.interpolate(p7_in, size=p6_in.shape[2:], mode='bilinear', align_corners=False)))

        # Weights for P5_0 and P6_1 to P5_1
        p5_w1 = F.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        # Connections for P5_0 and P6_1 to P5_1 respectively
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * F.interpolate(p6_up, size=p5_in.shape[2:], mode='bilinear', align_corners=False)))

        # Weights for P4_0 and P5_1 to P4_1
        p4_w1 = F.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for P4_0 and P5_1 to P4_1 respectively
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * F.interpolate(p5_up, size=p4_in.shape[2:], mode='bilinear', align_corners=False)))

        # Weights for P3_0 and P4_1 to P3_2
        p3_w1 = F.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for P3_0 and P4_1 to P3_2 respectively
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * F.interpolate(p4_up, size=p3_in.shape[2:], mode='bilinear', align_corners=False)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        p4_w2 = F.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights for P5_0, P5_1 and P4_2 to P5_2
        p5_w2 = F.relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights for P6_0, P6_1 and P5_2 to P6_2
        p6_w2 = F.relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p6_out = self.conv6_down(
            self.swish(weight[0] * p6_in + weight[1] * p6_up + weight[2] * self.p6_downsample(p5_out)))

        # Weights for P7_0 and P6_2 to P7_2
        p7_w2 = F.relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out