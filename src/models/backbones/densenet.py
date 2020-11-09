# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 16:34
# @Author : liumin
# @File : densenet.py

import torch
import torch.nn as nn
from torchvision.models.densenet import densenet121,densenet161,densenet169,densenet201


class Densenet(nn.Module):
    '''
        Densely Connected Convolutional Networks
        https://arxiv.org/pdf/1608.06993.pdf
    '''
    def __init__(self, backbone='densenet121', backbone_path=None, use_fpn=True):
        super(Densenet, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'densenet121':
            backbone = densenet121(pretrained=not backbone_path)
        elif backbone == 'densenet161':
            backbone = densenet161(pretrained=not backbone_path)
        elif backbone == 'densenet169':
            backbone = densenet169(pretrained=not backbone_path)
        elif backbone == 'densenet201':
            backbone = densenet201(pretrained=not backbone_path)

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        if self.use_fpn:
            self.out_channels = [256, 512, 1024]
        else:
            self.out_channels = [1024]

        self.conv1 = backbone.features.conv0
        self.norm0 = backbone.features.norm0
        self.relu0 = backbone.features.relu0
        self.pool0 = backbone.features.pool0
        self.layer1 = backbone.features.denseblock1
        self.trans_layer1 = backbone.features.transition1
        self.layer2 = backbone.features.denseblock2
        self.trans_layer2 = backbone.features.transition2
        self.layer3 = backbone.features.denseblock3
        self.trans_layer3 = backbone.features.transition3
        self.layer4 = backbone.features.denseblock4


    def forward(self, x):
        x = self.pool0(self.relu0(self.norm0(self.conv1(x))))
        x = self.layer1(x)
        out3 = self.layer2(self.trans_layer1(x))
        out4 = self.layer3(self.trans_layer2(out3))
        out5 = self.layer4(self.trans_layer3(out4))

        if self.use_fpn:
            return out3, out4, out5
        else:
            return out5

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model =Densenet('densenet121')
    print(model)