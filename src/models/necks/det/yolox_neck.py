# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 11:12
# @Author : liumin
# @File : yolox_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.models.bricks import ConvModule, DepthwiseSeparableConvModule
from src.models.modules.yolo_modules import CSPLayer
from src.models.necks.det.base_det_neck import BaseDetNeck


class YOLOXNeck(BaseDetNeck):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolox_s', # in_channels=[256, 512, 1024], out_channels=256,
                 layers=[3, 3, 3, 3], depthwise=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='Swish'),**kwargs):
        super(YOLOXNeck, self).__init__(**kwargs)
        assert isinstance(self.in_channels, list)
        self.subtype = subtype
        self.num_ins = len(self.in_channels)

        conv = DepthwiseSeparableConvModule if depthwise else ConvModule

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        self.out_channels = max(round(self.out_channels * width_mul), 1)
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # build top-down blocks
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(self.in_channels[idx], self.in_channels[idx - 1], 1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.top_down_blocks.append(
                CSPLayer(self.in_channels[idx - 1] * 2,self.in_channels[idx - 1],
                    n=layers[idx], shortcut=False, depthwise=depthwise,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

        # build bottom-up blocks
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(len(self.in_channels) - 1):
            self.downsamples.append(
                conv(self.in_channels[idx],self.in_channels[idx],3, stride=2, padding=1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.bottom_up_blocks.append(
                CSPLayer(self.in_channels[idx] * 2, self.in_channels[idx + 1], n=layers[idx],
                    shortcut=False,depthwise=depthwise,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.out_convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.out_convs.append(
                ConvModule(self.in_channels[i], self.out_channels, 1,
                    conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        assert len(x) == len(self.in_channels)

        # top-down path
        inner_outs = [x[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](torch.cat([upsample_feat, x[idx - 1]], 1))
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            downsample_feat = self.downsamples[idx](outs[-1])
            outs.append(self.bottom_up_blocks[idx](torch.cat([downsample_feat, inner_outs[idx + 1]], 1)))

        # out convs
        for idx, conv in enumerate(self.out_convs):
            outs[idx] = conv(outs[idx])

        return outs


if __name__ == "__main__":
    model = YOLOXNeck('yolox_s')
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''