# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/3 9:02
# @Author : liumin
# @File : resnet.py

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ResNet(nn.Module):
    '''
        Deep Residual Learning for Image Recognition
         <https://arxiv.org/pdf/1512.03385.pdf>
    '''
    def __init__(self, backbone='resnet50', backbone_path=None, use_fpn=True):
        super(ResNet, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not backbone_path)
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not backbone_path)
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not backbone_path)
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not backbone_path)
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not backbone_path)
        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        if self.use_fpn:
            self.out_channels = [512, 1024, 2048]
        else:
            self.out_channels = [2048]

        self.conv1 = nn.Sequential(list(backbone.children())[0])
        self.bn1 = nn.Sequential(list(backbone.children())[1])
        self.relu = nn.Sequential(list(backbone.children())[2])
        self.maxpool = nn.Sequential(list(backbone.children())[3])
        self.layer1 = nn.Sequential(list(backbone.children())[4])
        self.layer2 = nn.Sequential(list(backbone.children())[5])
        self.layer3 = nn.Sequential(list(backbone.children())[6])
        self.layer4 = nn.Sequential(list(backbone.children())[7])

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
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

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
