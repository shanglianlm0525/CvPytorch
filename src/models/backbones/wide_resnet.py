# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/9 10:48
# @Author : liumin
# @File : wide_resnet.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d, model_urls

"""
    Wide Residual Networks
    https://arxiv.org/pdf/1605.07146.pdf
"""


class WideResNet(nn.Module):

    def __init__(self, subtype='wide_resnet50_2', out_stages=[3, 5, 7], output_stride=16, classifier=False, num_classes=1000, pretrained = False, backbone_path=None):
        super(WideResNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if subtype == 'wide_resnet50_2':
            backbone = resnext50_32x4d(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif subtype == 'wide_resnet101_2':
            backbone = resnext101_32x8d(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.stem = nn.Sequential(*list(backbone.children())[0:4])
        self.layer1 = nn.Sequential(list(backbone.children())[4])
        self.layer2 = nn.Sequential(list(backbone.children())[5])
        self.layer3 = nn.Sequential(list(backbone.children())[6])
        self.layer4 = nn.Sequential(list(backbone.children())[7])

        if self.pretrained:
            self.load_pretrained_weights()
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
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)  if len(self.out_stages) > 1 else output[0]

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

    def load_pretrained_weights(self):
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__ == "__main__":
    model = WideResNet('wide_resnet50_2')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
