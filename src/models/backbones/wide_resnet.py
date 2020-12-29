# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/9 10:48
# @Author : liumin
# @File : wide_resnet.py

import torch
import torch.nn as nn
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d

"""
    Wide Residual Networks
    https://arxiv.org/pdf/1605.07146.pdf
"""

model_urls = {
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


class WideResNet(nn.Module):

    def __init__(self, name='wide_resnet50_2', out_stages=(1, 2, 3, 4), backbone_path=None):
        super(WideResNet, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        if name == 'wide_resnet50_2':
            backbone = resnext50_32x4d(pretrained=not self.backbone_path)
            self.out_channels = [256, 512, 1024, 2048]
        elif name == 'wide_resnet101_2':
            backbone = resnext101_32x8d(pretrained=not self.backbone_path)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(*list(backbone.children())[0:3])
        self.maxpool = nn.Sequential(list(backbone.children())[3])
        self.layer1 = nn.Sequential(list(backbone.children())[4])
        self.layer2 = nn.Sequential(list(backbone.children())[5])
        self.layer3 = nn.Sequential(list(backbone.children())[6])
        self.layer4 = nn.Sequential(list(backbone.children())[7])

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
        x = self.maxpool(x)
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


if __name__ == "__main__":
    model = WideResNet('wide_resnet50_2')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
