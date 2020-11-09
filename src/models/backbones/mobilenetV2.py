# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:16
# @Author : liumin
# @File : mobilenetV2.py

import torch
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2


class MobileNetV2(nn.Module):
    '''
        MobileNetV2: Inverted Residuals and Linear Bottlenecks
        https://arxiv.org/abs/1801.04381
    '''
    def __init__(self, backbone='mobilenet_v2', backbone_path=None, use_fpn=True):
        super(MobileNetV2, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'mobilenet_v2':
            backbone = mobilenet_v2(pretrained=not backbone_path)
            self.out_channels = [32, 96, 320]
        else:
            raise NotImplementedError

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.conv1 = nn.Sequential(list(backbone.features.children())[0])
        self.layer1 = nn.Sequential(list(backbone.features.children())[1])
        self.layer2 = nn.Sequential(*list(backbone.features.children())[2:4])
        self.layer3 = nn.Sequential(*list(backbone.features.children())[4:7])
        self.layer4 = nn.Sequential(*list(backbone.features.children())[7:11])
        self.layer5 = nn.Sequential(*list(backbone.features.children())[11:14])
        self.layer6 = nn.Sequential(*list(backbone.features.children())[14:17])
        self.layer7 = nn.Sequential(list(backbone.features.children())[17])


    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        out3 = self.layer3(self.layer2(x))
        out4 = self.layer5(self.layer4(out3))
        out5 = self.layer7(self.layer6(out4))

        if self.use_fpn:
            return out3, out4, out5
        else:
            return out5

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model =MobileNetV2('mobilenet_v2')
    print(model)

