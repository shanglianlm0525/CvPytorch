# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/20 17:14
# @Author : liumin
# @File : csp_darknet.py


import torch
import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm

from src.models.bricks import ConvModule, DepthwiseSeparableConvModule
from src.models.modules.yolo_modules import Focus, CSPLayer, SPPF

"""
    YOLOv4: Optimal Speed and Accuracy of Object Detection
    https://arxiv.org/pdf/2004.10934.pdf
    YOLOX: Exceeding YOLO Series in 2021
    https://arxiv.org/pdf/2107.08430.pdf
    AIRDet:
    https://github.com/tinyvision/AIRDet
"""

class CSPDarknet(nn.Module):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='cspdark_s', out_channels=[64, 128, 256, 512, 1024], layers=[3, 9, 9, 3], spp_ksizes=(5, 9, 13), depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='Swish'),
                 out_stages=[2, 3, 4], output_stride=32, backbone_path=None, pretrained=False,
                 frozen_stages=-1, norm_eval=False):
        super(CSPDarknet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained
        self.out_channels = out_channels
        self.layers = layers
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        conv = DepthwiseSeparableConvModule if depthwise else ConvModule

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.out_channels = list(map(lambda x: int(x * width_mul), out_channels))
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))

        # stem
        self.stem = Focus(3, self.out_channels[0], kernel_sizes=3, conv_cfg=conv_cfg,
            norm_cfg=norm_cfg, act_cfg=act_cfg)

        # dark1
        self.stage1 = nn.Sequential(
            conv(self.out_channels[0], self.out_channels[1], 3, 2, padding=1,
                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            CSPLayer(self.out_channels[1], self.out_channels[1], n=layers[0], shortcut=True, depthwise=depthwise,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # dark2
        self.stage2 = nn.Sequential(
            conv(self.out_channels[1], self.out_channels[2], 3, 2, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            CSPLayer(self.out_channels[2], self.out_channels[2], n=layers[1], shortcut=True, depthwise=depthwise,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # dark3
        self.stage3 = nn.Sequential(
            conv(self.out_channels[2], self.out_channels[3], 3, 2, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            CSPLayer(self.out_channels[3], self.out_channels[3], n=layers[2], shortcut=True, depthwise=depthwise,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        # dark4
        self.stage4 = nn.Sequential(
            conv(self.out_channels[3], self.out_channels[4], 3, 2, padding=1,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            SPPF(self.out_channels[4], self.out_channels[4], kernel_sizes=spp_ksizes,
                          conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
            CSPLayer(self.out_channels[4], self.out_channels[4], n=layers[3], shortcut=False, depthwise=depthwise,
                     conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        )

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == "__main__":
    model = CSPDarknet('cspdark_s')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)