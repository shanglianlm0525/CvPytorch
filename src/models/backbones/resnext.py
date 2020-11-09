# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/9 10:46
# @Author : liumin
# @File : resnext.py

import torch
import torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d


class ResNeXt(nn.Module):
    '''
        Aggregated Residual Transformation for Deep Neural Networks
        https://arxiv.org/pdf/1611.05431.pdf
    '''
    def __init__(self, backbone='resnext50_32x4d', backbone_path=None, use_fpn=True):
        super(ResNeXt, self).__init__()
        self.use_fpn = use_fpn

        self.out_channels = [512, 1024, 2048]
        if backbone == 'resnext50_32x4d':
            backbone = resnext50_32x4d(pretrained=not backbone_path)
        elif backbone == 'resnext101_32x8d':
            backbone = resnext101_32x8d(pretrained=not backbone_path)
        else:
            raise NotImplementedError

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

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

if __name__=="__main__":
    model =ResNeXt('resnext50_32x4d')
    print(model)