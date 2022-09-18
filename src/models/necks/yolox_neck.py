# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/27 14:56
# @Author : liumin
# @File : yolox_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.yolox_modules import BaseConv, CSPLayer


class YOLOXNeck(nn.Module):
    def __init__(self, subtype='yolox_s', channels=[256, 512, 1024], depth_mul=1.0, width_mul=1.0):
        super(YOLOXNeck, self).__init__()
        assert isinstance(channels, list)
        self.subtype = subtype
        self.channels = channels
        self.num_ins = len(channels)

        act = "silu"
        # depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        channels = list(map(lambda x: max(round(x * width_mul), 1), channels))
        layers = list(map(lambda x: max(round(x * depth_mul), 1), [3, 3, 3, 3]))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(channels[2], channels[1], 1, 1, act=act)

        self.C3_p4 = CSPLayer(2 * channels[1],channels[1], layers[0],False,act=act)  # cat

        self.reduce_conv1 = BaseConv(channels[1], channels[0], 1, 1, act=act)

        self.C3_p3 = CSPLayer(2 * channels[0], channels[0], layers[1], False, act=act)

        # bottom-up conv
        self.bu_conv2 = BaseConv(channels[0], channels[0], 3, 2, act=act)
        self.C3_n3 = CSPLayer(2 * channels[0], channels[1], layers[2], False, act=act)

        # bottom-up conv
        self.bu_conv1 = BaseConv(channels[1] , channels[1], 3, 2, act=act)
        self.C3_n4 = CSPLayer(2 * channels[1], channels[2], layers[3],False,act=act)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        assert len(x) == len(self.channels)
        [x2, x1, x0] = x

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs