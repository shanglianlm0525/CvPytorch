# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 16:34
# @Author : liumin
# @File : densenet.py

import torch
import torch.nn as nn
from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

"""
    Densely Connected Convolutional Networks
    https://arxiv.org/pdf/1608.06993.pdf
"""


class Densenet(nn.Module):

    def __init__(self, subtype='densenet121', out_stages=[2, 3, 4], backbone_path=None):
        super(Densenet, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        self.out_channels = [128, 256, 512, 1024]
        if subtype == 'densenet121':
            features = densenet121(pretrained=not backbone_path).features
        elif subtype == 'densenet161':
            features = densenet161(pretrained=not backbone_path).features
        elif subtype == 'densenet169':
            features = densenet169(pretrained=not backbone_path).features
        elif subtype == 'densenet201':
            features = densenet201(pretrained=not backbone_path).features

        self.conv1 = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0
        )
        self.layer1 = nn.Sequential(
            features.denseblock1,
            features.transition1
        )
        self.layer2 = nn.Sequential(
            features.denseblock2,
            features.transition2
        )
        self.layer3 = nn.Sequential(
            features.denseblock3,
            features.transition3
        )
        self.layer4 = features.denseblock4

        self.init_weights()

        if self.backbone_path:
            self.features.load_state_dict(torch.load(self.backbone_path))

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

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    model = Densenet('densenet121')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
