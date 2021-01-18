# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/8 16:31
# @Author : liumin
# @File : yolov5_backbone.py

import torch
import torch.nn as nn
# from ..modules.yolov5_modules import parse_yolov5_model
from CvPytorch.src.models.modules.yolov5_modules import parse_yolov5_model


class YOLOv5Backbone(nn.Module):
                # [from, number, module, args]
    model_cfg = [[-1, 1, 'Focus', [64, 3]],  # 0-P1/2
                 [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                 [-1, 3, 'BottleneckCSP', [128]],
                 [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                 [-1, 9, 'BottleneckCSP', [256]],
                 [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                 [-1, 9, 'BottleneckCSP', [512]],
                 [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                 [-1, 1, 'SPP', [1024, [5, 9, 13]]],
                 [-1, 3, 'BottleneckCSP', [1024, False]]  # 9
                 ]
    def __init__(self, subtype='yolov5s', out_stages=[2,3,4], backbone_path=None, num_outs=18):
        super(YOLOv5Backbone, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path
        #  num_outs = 18 # na * (nc + 5)  # number of outputs = anchors * (classes + 5)

        if subtype == 'yolov5s':
            depth_mul = 0.33  # model depth multiple
            width_mul = 0.50  # layer channel multiple
            backbone = parse_yolov5_model(self.model_cfg, depth_mul, width_mul, num_outs)
            self.out_channels = [64, 128, 256, 512]
        elif subtype == 'yolov5m':
            depth_mul = 0.67  # model depth multiple
            width_mul = 0.75  # layer channel multiple
            backbone = parse_yolov5_model(self.model_cfg, depth_mul, width_mul, num_outs)
            self.out_channels = [96, 192, 384, 768]
        elif subtype == 'yolov5l':
            depth_mul = 1.0  # model depth multiple
            width_mul = 1.0  # layer channel multiple
            backbone = parse_yolov5_model(self.model_cfg, depth_mul, width_mul, num_outs)
            self.out_channels = [128, 256, 512, 1024]
        elif subtype == 'yolov5x':
            depth_mul = 1.33  # model depth multiple
            width_mul = 1.25  # layer channel multiple
            backbone = parse_yolov5_model(self.model_cfg, depth_mul, width_mul, num_outs)
            self.out_channels = [160, 320, 640, 1280]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(list(backbone.children())[0])
        self.layer1 = nn.Sequential(*list(backbone.children())[1:3])
        self.layer2 = nn.Sequential(*list(backbone.children())[3:5])
        self.layer3 = nn.Sequential(*list(backbone.children())[5:7])
        self.layer4 = nn.Sequential(*list(backbone.children())[7:])

        self.init_weights()

        if self.backbone_path:
            self.backbone.load_state_dict(torch.load(self.backbone_path))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

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
    model = YOLOv5Backbone('yolov5s')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)