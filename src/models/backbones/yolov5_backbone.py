# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/29 14:11
# @Author : liumin
# @File : yolov5_backbone.py

import torch
import torch.nn as nn
from src.models.modules.yolov5_modules import Conv, C3, SPPF, SPP


class YOLOv5Backbone(nn.Module):
    def __init__(self, subtype='yolov5s', out_stages=[2,3,4], output_stride=32, layers=[3, 6, 9, 3], depth_mul=1.0, width_mul=1.0, backbone_path=None, pretrained = False):
        super(YOLOv5Backbone, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.width_mul = width_mul
        self.depth_mul = depth_mul
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        self.out_channels = [64, 128, 256, 512, 1024]
        self.out_channels = in_places = list(map(lambda x: int(x * self.width_mul), self.out_channels))
        layers = list(map(lambda x: max(round(x * self.depth_mul), 1), layers))

        self.conv1 = Conv(3, in_places[0], k=(6, 6), s=(2, 2), p=(2, 2))
        self.layer1 = nn.Sequential(Conv(in_places[0], in_places[1], 3, 2),
                                    C3(in_places[1], in_places[1], layers[0]))
        self.layer2 = nn.Sequential(Conv(in_places[1], in_places[2], 3, 2),
                                    C3(in_places[2], in_places[2], layers[1]))
        self.layer3 = nn.Sequential(Conv(in_places[2], in_places[3], 3, 2),
                                    C3(in_places[3], in_places[3], layers[2]))
        self.layer4 = nn.Sequential(Conv(in_places[3], in_places[4], 3, 2),
                                    SPP(in_places[4], in_places[4], [5, 9, 13]))

        self.out_channels = [int(otc * width_mul) for otc in self.out_channels]
        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def forward(self, x):
        x = self.conv1(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)
        return output


if __name__ == "__main__":
    model = YOLOv5Backbone(subtype='yolov5x', depth_mul = 0.33, width_mul = 0.50)
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)