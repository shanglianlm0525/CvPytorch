# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/15 14:04
# @Author : liumin
# @File : mobilenet_v3.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo
from torchvision.models.mobilenet import mobilenet_v3_small, mobilenet_v3_large
from torchvision.models.mobilenetv3 import model_urls

"""
    Searching for MobileNetV3
    https://arxiv.org/pdf/1905.02244.pdf
"""


class MobileNetV3(nn.Module):

    def __init__(self, subtype='mobilenet_v3_small', out_stages=[3, 4, 5], output_stride=32, classifier=False, num_classes=1000, backbone_path=None, pretrained = True):
        super(MobileNetV3, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'mobilenet_v3_small':
            features = mobilenet_v3_small(self.pretrained).features

            mb3s = list(features.children())
            self.stem = nn.Sequential(mb3s[0])  # x2
            self.stage1 = nn.Sequential(mb3s[1])
            self.stage2 = nn.Sequential(*mb3s[2:4])
            self.stage3 = nn.Sequential(*mb3s[4:7])
            self.stage4 = nn.Sequential(*mb3s[7:9])
            self.stage5 = nn.Sequential(*mb3s[9:12])

            if self.classifier:
                self.last_conv = nn.Sequential(list(features.children())[12])
                self.fc = mobilenet_v3_small(self.pretrained).classifier
                self.fc[3] = nn.Linear(self.fc[3].in_features, self.num_classes)
                self.out_channels = self.num_classes

            self.out_channels = [16, 16, 24, 40, 48, 96]
        elif self.subtype == 'mobilenet_v3_large':
            features = mobilenet_v3_large(self.pretrained).features

            mb3l = list(features.children())
            self.stem = nn.Sequential(mb3l[0])  # x2
            self.stage1 = nn.Sequential(mb3l[1])
            self.stage2 = nn.Sequential(*mb3l[2:4])
            self.stage3 = nn.Sequential(*mb3l[4:7])
            self.stage4 = nn.Sequential(*mb3l[7:11])
            self.stage5 = nn.Sequential(*mb3l[11:13])
            self.stage6 = nn.Sequential(*mb3l[13:16])

            if self.classifier:
                self.last_conv = nn.Sequential(list(features.children())[16])
                self.fc = mobilenet_v3_large(self.pretrained).classifier
                self.fc[3] = nn.Linear(self.fc[3].in_features, self.num_classes)
                self.out_channels = self.num_classes

            self.out_channels = [16, 16, 24, 40, 80, 112, 160]
        else:
            raise NotImplementedError

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

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
                m.momentum = 0.1
                m.eps = 1e-05
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.stem(x)
        output = []
        if self.subtype == 'mobilenet_v3_small':
            for i in range(1, 6):
                stage = getattr(self, 'stage{}'.format(i))
                x = stage(x)
                if i in self.out_stages and not self.classifier:
                    output.append(x)
        else:
            for i in range(1, 7):
                stage = getattr(self, 'stage{}'.format(i))
                x = stage(x)
                if i in self.out_stages and not self.classifier:
                    output.append(x)
        if self.classifier:
            x = self.last_conv(x)
            x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]


    def load_pretrained_weights(self):
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__=="__main__":
    model =MobileNetV3('mobilenet_v3_small')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)