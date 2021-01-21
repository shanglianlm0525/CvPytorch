# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 16:13
# @Author : liumin
# @File : vgg.py

import torch
import torch.nn as nn
from torchvision.models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

"""
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/pdf/1409.1556.pdf
"""

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, subtype='vgg16', out_stages=[2,3,4], backbone_path=None):
        super(VGG, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        if subtype == 'vgg11':
            features = vgg11_bn(pretrained=not self.backbone_path).features
            self.out_channels = [128, 256, 512, 512]
        elif subtype == 'vgg13':
            features = vgg13_bn(pretrained=not self.backbone_path).features
            self.out_channels = [128, 256, 512, 512]
        elif subtype == 'vgg16':
            features = vgg16_bn(pretrained=not self.backbone_path).features
            self.out_channels = [128, 256, 512, 512]
        elif subtype == 'vgg19':
            features = vgg19_bn(pretrained=not self.backbone_path).features
            self.out_channels = [128, 256, 512, 512]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(*list(features.children())[:7])
        self.layer1 = nn.Sequential(*list(features.children())[7:14])
        self.layer2 = nn.Sequential(*list(features.children())[14:24])
        self.layer3 = nn.Sequential(*list(features.children())[24:34])
        self.layer4 = nn.Sequential(*list(features.children())[34:43])

        if self.backbone_path:
            self.features.load_state_dict(torch.load(self.backbone_path))
        else:
            self.init_weights()

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


if __name__=="__main__":
    model =VGG('vgg16')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)