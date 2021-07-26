# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 10:35
# @Author : liumin
# @File : csp_darknet.py

import torch
import torch.nn as nn

from src.models.modules.yolox_modules import BaseConv, DWConv, Focus, CSPLayer, SPPBottleneck

"""
    YOLOX: Exceeding YOLO Series in 2021
    https://arxiv.org/pdf/2107.08430.pdf
"""


class CspDarkNet(nn.Module):

    def __init__(self, subtype='yolox_s', out_stages=[2,3,4], output_stride=32, depthwise = False, backbone_path=None, pretrained=False):
        super(CspDarkNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        Conv = DWConv if depthwise else BaseConv

        if self.subtype == 'yolox_s':
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = [32, 64, 128, 256, 512]
        elif self.subtype == 'yolox_m':
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = [48, 96, 192, 384, 768]
        elif self.subtype == 'yolox_l':
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = [64, 128, 256, 512, 1024]
        elif self.subtype == 'yolox_x':
            dep_mul, wid_mul = 1.33, 1.25
            self.out_channels = [80, 160, 320, 640, 1280]
        else:
            raise NotImplementedError

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3)

        # dark2
        self.stage1 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise),
        )

        # dark3
        self.stage2 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise),
        )

        # dark4
        self.stage3 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise),
        )

        # dark5
        self.stage4 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPPBottleneck(base_channels * 16, base_channels * 16),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise),
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
        return tuple(output) if len(self.out_stages) > 1 else output[0]


    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = CspDarkNet('yolox_l')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)