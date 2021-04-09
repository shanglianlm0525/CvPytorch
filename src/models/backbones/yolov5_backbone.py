# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/8 16:31
# @Author : liumin
# @File : yolov5_backbone.py

import torch
import torch.nn as nn
# from ..modules.yolov5_modules import parse_yolov5_model
from ..modules.yolov5_modules import Focus, Conv, C3, SPP


class YOLOv5Backbone(nn.Module):
    def __init__(self, subtype='yolov5s', out_stages=[2,3,4], output_stride = 32, backbone_path=None, pretrained = False):
        super(YOLOv5Backbone, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'yolov5s':
            self.depth_mul = 0.33  # model depth multiple
            self.width_mul = 0.50  # layer channel multiple
            self.out_channels = [32, 64, 128, 256, 512]
        elif self.subtype == 'yolov5m':
            self.depth_mul = 0.67  # model depth multiple
            self.width_mul = 0.75  # layer channel multiple
            self.out_channels = [48, 96, 192, 384, 768]
        elif self.subtype == 'yolov5l':
            self.depth_mul = 1.0  # model depth multiple
            self.width_mul = 1.0  # layer channel multiple
            self.out_channels = [64, 128, 256, 512, 1024]
        elif self.subtype == 'yolov5x':
            self.depth_mul = 1.33  # model depth multiple
            self.width_mul = 1.25  # layer channel multiple
            self.out_channels = [80, 160, 320, 640, 1280]
        else:
            raise NotImplementedError

        in_places = list(map(lambda x: int(x * self.width_mul), [64, 128, 256, 512, 1024]))
        layers = list(map(lambda x: max(round(x * self.depth_mul), 1), [3, 9, 9]))

        self.conv1 = Focus(3, in_places[0], 3)
        self.layer1 = nn.Sequential(Conv(in_places[0], in_places[1], 3, 2),
                                    C3(in_places[1], in_places[1], layers[0]))
        self.layer2 = nn.Sequential(Conv(in_places[1], in_places[2], 3, 2),
                                    C3(in_places[2], in_places[2], layers[1]))
        self.layer3 = nn.Sequential(Conv(in_places[2], in_places[3], 3, 2),
                                    C3(in_places[3], in_places[3], layers[2]))
        self.layer4 = nn.Sequential(Conv(in_places[3], in_places[4], 3, 2),
                                    SPP(in_places[4], in_places[4], [5, 9, 13]))

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()
        if self.pretrained:
            self.load_pretrained_weights()


    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x):
        x = self.conv1(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


if __name__ == "__main__":
    model = YOLOv5Backbone('yolov5x')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)