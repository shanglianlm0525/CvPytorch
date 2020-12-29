# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:16
# @Author : liumin
# @File : mobilenet_v2.py

import torch
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2

"""
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

class MobileNetV2(nn.Module):

    def __init__(self, name='mobilenet_v2', out_stages=(2, 3, 5, 7), backbone_path=None):
        super(MobileNetV2, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        if name == 'mobilenet_v2':
            features = mobilenet_v2(pretrained=not self.backbone_path).features
            self.out_channels = [16, 24, 32, 64, 96, 160, 320]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(list(features.children())[0])
        self.stage1 = nn.Sequential(list(features.children())[1])
        self.stage2 = nn.Sequential(*list(features.children())[2:4])
        self.stage3 = nn.Sequential(*list(features.children())[4:7])
        self.stage4 = nn.Sequential(*list(features.children())[7:11])
        self.stage5 = nn.Sequential(*list(features.children())[11:14])
        self.stage6 = nn.Sequential(*list(features.children())[14:17])
        self.stage7 = nn.Sequential(list(features.children())[17])

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.conv1(x)
        output = []
        for i in range(1, 8):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model =MobileNetV2('mobilenet_v2')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)