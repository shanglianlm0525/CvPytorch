# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/21 10:12
# @Author : liumin
# @File : sgcpnet_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            nn.Hardsigmoid())

    def forward(self, x):
        return x * self.se(x)


class InvertedResidualBlock(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_size),)

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class SGCPNetBackbone(nn.Module):

    def __init__(self, subtype='', out_stages=[3, 4, 5], output_stride=16, classifier=False, backbone_path=None, pretrained = False):
        super(SGCPNetBackbone, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(16),
                                nn.Hardswish())

        self.stage1 = InvertedResidualBlock(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2)

        self.stage2 = nn.Sequential(
            InvertedResidualBlock(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            InvertedResidualBlock(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1))

        self.stage3 = nn.Sequential(
            InvertedResidualBlock(5, 24, 96, 40, nn.Hardswish(), SeModule(40), 2),
            InvertedResidualBlock(5, 40, 240, 40, nn.Hardswish(), SeModule(40), 1),
            InvertedResidualBlock(5, 40, 240, 40, nn.Hardswish(), SeModule(40), 1),
            InvertedResidualBlock(5, 40, 120, 48, nn.Hardswish(), SeModule(48), 1),
            InvertedResidualBlock(5, 48, 144, 48, nn.Hardswish(), SeModule(48), 1))

        self.stage4 = nn.Sequential(
            InvertedResidualBlock(5, 48, 288, 96, nn.Hardswish(), SeModule(96), 2),
            InvertedResidualBlock(5, 96, 576, 96, nn.Hardswish(), SeModule(96), 1),
            InvertedResidualBlock(5, 96, 576, 96, nn.Hardswish(), SeModule(96), 1))

        self.out_channels = [16, 16, 24, 48, 96]

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        self.init_weights()
        if self.pretrained:
           self.load_pretrained_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]


    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__=="__main__":
    model = SGCPNetBackbone('')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)