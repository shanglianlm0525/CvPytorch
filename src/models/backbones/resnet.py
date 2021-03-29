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

    def __init__(self, subtype='resnet50', out_stages=[2, 3, 4], output_stride = 32, backbone_path=None, pretrained = False):
        super(ResNet, self).__init__()
        self.out_stages = out_stages
        self.output_stride = output_stride # 8, 16, 32
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if subtype == 'resnet18':
            self.pretrained = True
            backbone = resnet18(pretrained=self.pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif subtype == 'resnet34':
            self.pretrained = True
            backbone = resnet34(pretrained=self.pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif subtype == 'resnet50':
            self.pretrained = True
            backbone = resnet50(pretrained=self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif subtype == 'resnet101':
            self.pretrained = True
            backbone = resnet101(pretrained=self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif subtype == 'resnet152':
            self.pretrained = True
            backbone = resnet152(pretrained=self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1]+1]

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if self.output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif self.output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (subtype == 'resnet34' or subtype == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        if self.output_stride == 8 or self.output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv1' in n and (subtype == 'resnet34' or subtype == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
                elif 'downsample.0' in n:
                    m.stride = (s4, s4)

        if not self.pretrained:
            if self.backbone_path:
                self.pretrained = True
                self.backbone.load_state_dict(torch.load(self.backbone_path))
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
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
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