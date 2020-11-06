# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 16:13
# @Author : liumin
# @File : vgg.py

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


class VGG(nn.Module):
    '''
        Very Deep Convolutional Networks for Large-Scale Image Recognition
        https://arxiv.org/pdf/1409.1556.pdf
    '''
    def __init__(self, backbone='vgg16', backbone_path=None, use_fpn=True):
        super(VGG, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'vgg11':
            backbone = vgg11_bn(pretrained=not backbone_path)
        elif backbone == 'vgg13':
            backbone = vgg13_bn(pretrained=not backbone_path)
        elif backbone == 'vgg16':
            backbone = vgg16_bn(pretrained=not backbone_path)
        elif backbone == 'vgg19':
            backbone = vgg19_bn(pretrained=not backbone_path)

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        if self.use_fpn:
            self.out_channels = [256, 512, 512]
        else:
            self.out_channels = [512]

        self.conv1 = nn.Sequential(*list(backbone.features.children())[:7])
        self.layer1 = nn.Sequential(*list(backbone.features.children())[7:14])
        self.layer2 = nn.Sequential(*list(backbone.features.children())[14:24])
        self.layer3 = nn.Sequential(*list(backbone.features.children())[24:34])
        self.layer4 = nn.Sequential(*list(backbone.features.children())[34:43])

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        out3 = self.layer2(x)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        if self.use_fpn:
            return out3, out4, out5
        else:
            return out5

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model =VGG('vgg16')
    print(model)