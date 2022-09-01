# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/11 15:01
# @Author : liumin
# @File : yolov7_backbone.py

import torch
import torch.nn as nn

from src.models.modules.yolov7_modules import Conv, EELAN, DownA, SPPCSPC


class YOLOv7Backbone(nn.Module):
    def __init__(self, subtype='yolov7s', out_stages=[3, 4, 5], output_stride=32, depth_mul=1.0, width_mul=1.0, backbone_path=None, pretrained = False):
        super(YOLOv7Backbone, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        in_places = [32, 64, 128, 256, 512, 1024]
        out_channels = [32, 64, 256, 512, 1024, 512]
        in_places = list(map(lambda x: int(x * width_mul), in_places))
        self.out_channels = list(map(lambda x: int(x * width_mul), out_channels))

        self.stem = Conv(3, in_places[0], k=3, s=1, p=1)  # 0
        self.layer1 = nn.Sequential(Conv(in_places[0], in_places[1], 3, 2),
                                    Conv(in_places[1], in_places[1], 3, 1))
        self.layer2 = nn.Sequential(Conv(in_places[1], in_places[2], 3, 2),
                                    EELAN(in_places[2], in_places[1], in_places[3]))

        self.layer3 = nn.Sequential(DownA(in_places[3], in_places[2]),
                                    EELAN(in_places[3], in_places[2], in_places[4]))

        self.layer4 = nn.Sequential(DownA(in_places[4], in_places[3]),
                                    EELAN(in_places[4], in_places[3], in_places[5]))

        self.layer5 = nn.Sequential(DownA(in_places[5], in_places[4]),
                                    EELAN(in_places[5], in_places[3], in_places[5]))

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
        x = self.stem(x)
        output = []
        for i in range(1, 6):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)
        return output


if __name__ == "__main__":
    model = YOLOv7Backbone(subtype='yolov7l', depth_mul= 1.0, width_mul= 1.0)
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)