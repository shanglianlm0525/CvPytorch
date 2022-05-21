# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/9 10:46
# @Author : liumin
# @File : resnext.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d, model_urls

"""
    Aggregated Residual Transformation for Deep Neural Networks
    https://arxiv.org/pdf/1611.05431.pdf
"""


class ResNeXt(nn.Module):

    def __init__(self, subtype='resnext50_32x4d', out_stages=[2, 3, 4], output_stride=32, classifier=False, num_classes=1000, backbone_path=None, pretrained = True):
        super(ResNeXt, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'resnext50_32x4d':
            resnext = resnext50_32x4d(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif self.subtype == 'resnext101_32x8d':
            resnext = resnext101_32x8d(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.stem = nn.Sequential(*list(resnext.children())[:4])
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        if self.classifier:
            self.avgpool = resnext.avgpool
            self.fc = nn.Linear(resnext.fc.in_features, self.num_classes)
            self.out_channels = self.num_classes

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
            if i in self.out_stages and not self.classifier:
                output.append(x)

        if self.classifier:
            x = self.avgpool(x)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]

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
    model = ResNeXt('resnext50_32x4d')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
