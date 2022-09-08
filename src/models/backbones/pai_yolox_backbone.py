# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/6 10:23
# @Author : liumin
# @File : pai_yolox_backbone.py

import torch
import torch.nn as nn
from src.models.modules.yolox_modules import RepVGGBlock, MT_SPPF


class PAI_YOLOXBackbone(nn.Module):
    def __init__(self, subtype='pai_yolox_s', out_stages=[2, 3, 4], output_stride=32, depth_mul=1.0, width_mul=1.0, backbone_path=None,
                 pretrained=False):
        super(PAI_YOLOXBackbone, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        out_channels = [64, 128, 256, 512, 1024]
        layers = [6, 12, 18, 6]
        self.out_channels = list(map(lambda x: int(x * width_mul), out_channels))
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))

        # stem
        self.stem = RepVGGBlock(in_channels=3, out_channels=self.out_channels[0], ksize=3, stride=2)

        self.stage1 = self._make_stage(self.out_channels[0], self.out_channels[1], layers[0])
        self.stage2 = self._make_stage(self.out_channels[1], self.out_channels[2], layers[1])
        self.stage3 = self._make_stage(self.out_channels[2], self.out_channels[3], layers[2])
        self.stage4 = self._make_stage(self.out_channels[3], self.out_channels[4], layers[3], add_ppf=True)

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()

    def _make_stage(self, in_channels, out_channels, layers, stride=2, add_ppf=False):
        blocks = []
        blocks.append(RepVGGBlock(in_channels, out_channels, ksize=3, stride=stride))
        for i in range(layers):
            blocks.append(RepVGGBlock(out_channels, out_channels))
        if add_ppf:
            blocks.append(MT_SPPF(out_channels, out_channels, kernel_size=5))
        return nn.Sequential(*blocks)

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
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


if __name__ == "__main__":
    model = PAIYOLOXBackbone('cspdark_s')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)