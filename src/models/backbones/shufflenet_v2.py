# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:36
# @Author : liumin
# @File : shufflenet_v2.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.shufflenetv2 import model_urls

"""
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""


class ShuffleNetV2(nn.Module):

    def __init__(self, subtype='shufflenetv2_x1.0', out_stages=[2, 3, 4], output_stride=32, classifier=False, num_classes=1000, pretrained=True, backbone_path=None):
        super(ShuffleNetV2, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if self.subtype == 'shufflenetv2_x0.5':
            backbone = shufflenet_v2_x0_5(self.pretrained)
            self.out_channels = [24, 48, 96, 192, 1024]
        elif self.subtype == 'shufflenetv2_x1.0':
            backbone = shufflenet_v2_x1_0(self.pretrained)
            self.out_channels = [24, 116, 232, 464, 1024]
        elif self.subtype == 'shufflenetv2_x1.5':
            backbone = shufflenet_v2_x1_5(self.pretrained)
            self.out_channels = [24, 176, 352, 704, 1024]
        elif self.subtype == 'shufflenetv2_x2.0':
            backbone = shufflenet_v2_x2_0(self.pretrained)
            self.out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        self.stem = backbone.conv1
        self.maxpool = backbone.maxpool
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        if self.classifier:
            self.conv5 = backbone.conv5
            self.fc = nn.Linear(backbone.fc.in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained and self.backbone_path is not None:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, 'layer{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        if self.classifier:
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]


    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__=="__main__":
    model =ShuffleNetV2('shufflenetv2_x0.5', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)