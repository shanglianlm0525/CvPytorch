# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:58
# @Author : liumin
# @File : squeezenet.py

import torch
import torch.nn as nn
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1


class SqueezeNet(nn.Module):
    '''
        SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
        https://arxiv.org/abs/1602.07360
    '''
    def __init__(self, backbone='squeezenet1_1', backbone_path=None, use_fpn=True):
        super(SqueezeNet, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'squeezenet1_0':
            backbone = squeezenet1_0(pretrained=not backbone_path)
            self.out_channels = [64, 128, 256]
        elif backbone == 'squeezenet1_1':
            backbone = squeezenet1_1(pretrained=not backbone_path)
            self.out_channels = [64, 128, 256]
        else:
            raise NotImplementedError

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.conv1 = nn.Sequential(*list(backbone.features.children())[0:2])
        self.layer1 = nn.Sequential(*list(backbone.features.children())[2:5])
        self.layer2 = nn.Sequential(*list(backbone.features.children())[5:8])
        self.layer3 = nn.Sequential(*list(backbone.features.children())[8:13])

    def forward(self, x):
        x = self.conv1(x)
        out3 = self.layer1(x)
        out4 = self.layer2(out3)
        out5 = self.layer3(out4)

        if self.use_fpn:
            return out3, out4, out5
        else:
            return out5

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model = SqueezeNet('squeezenet1_1')
    print(model)

