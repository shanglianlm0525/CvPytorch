# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/7 9:39
# @Author : liumin
# @File : csp_darknet.py

import torch
import torch.nn as nn

from src.models.modules.yolox_modules import Focus, BaseConv, CSPLayer, SPPBottleneck

"""
    YOLOv4: Optimal Speed and Accuracy of Object Detection
    https://arxiv.org/pdf/2004.10934.pdf
    YOLOX: Exceeding YOLO Series in 2021
    https://arxiv.org/pdf/2107.08430.pdf
"""

class CspDarkNet(nn.Module):
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='cspdark_s', out_stages=[2, 3, 4], output_stride=32, backbone_path=None,
                 pretrained=False):
        super(CspDarkNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        '''
        if self.subtype == 'cspdark_s':
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = [32, 64, 128, 256, 512]
        elif self.subtype == 'cspdark_m':
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = [48, 96, 192, 384, 768]
        elif self.subtype == 'cspdark_l':
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = [64, 128, 256, 512, 1024]
        elif self.subtype == 'cspdark_x':
            dep_mul, wid_mul = 1.33, 1.25
            self.out_channels = [80, 160, 320, 640, 1280]
        else:
            raise NotImplementedError
        '''
        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.out_channels = [64, 128, 256, 512, 1024]
        layers = [3, 9, 9, 3]
        self.out_channels = in_places = list(map(lambda x: int(x * width_mul), self.out_channels))
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))

        act = "silu"
        # stem
        self.stem = Focus(3, in_places[0], ksize=3, act=act)

        # dark2
        self.stage1 = nn.Sequential(
            BaseConv(in_places[0], in_places[1], 3, 2, act=act),
            CSPLayer(in_places[1], in_places[1], n=layers[0], act=act),
        )

        # dark3
        self.stage2 = nn.Sequential(
            BaseConv(in_places[1], in_places[2], 3, 2, act=act),
            CSPLayer(in_places[2], in_places[2], n=layers[1], act=act),
        )

        # dark4
        self.stage3 = nn.Sequential(
            BaseConv(in_places[2], in_places[3], 3, 2),
            CSPLayer(in_places[3], in_places[3], n=layers[2]),
        )

        # dark5
        self.stage4 = nn.Sequential(
            BaseConv(in_places[3], in_places[4], 3, 2, act=act),
            SPPBottleneck(in_places[4], in_places[4], act=act),
            CSPLayer(in_places[4], in_places[4], n=layers[3], shortcut=False, act=act),
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
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


if __name__ == "__main__":
    model = CspDarkNet('cspdark_s')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)