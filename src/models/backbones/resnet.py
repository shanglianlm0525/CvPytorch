# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/3 9:02
# @Author : liumin
# @File : resnet.py

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

"""
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
"""

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet(nn.Module):

    def __init__(self, subtype='resnet50', out_stages=[2, 3, 4], backbone_path=None):
        super(ResNet, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        if subtype == 'resnet18':
            backbone = resnet18(pretrained=not self.backbone_path)
            self.out_channels = [64, 128, 256, 512]
        elif subtype == 'resnet34':
            backbone = resnet34(pretrained=not self.backbone_path)
            self.out_channels = [64, 128, 256, 512]
        elif subtype == 'resnet50':
            backbone = resnet50(pretrained=not self.backbone_path)
            self.out_channels = [256, 512, 1024, 2048]
        elif subtype == 'resnet101':
            backbone = resnet101(pretrained=not self.backbone_path)
            self.out_channels = [256, 512, 1024, 2048]
        elif subtype == 'resnet152':
            backbone = resnet152(pretrained=not self.backbone_path)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(list(backbone.children())[0])
        self.bn1 = nn.Sequential(list(backbone.children())[1])
        self.act1 = nn.Sequential(list(backbone.children())[2])
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
    model = ResNet('resnet50')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)