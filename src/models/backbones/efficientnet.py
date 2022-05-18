# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/28 14:02
# @Author : liumin
# @File : efficientnet.py

import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models as models
from torchvision.models.efficientnet import model_urls
'''
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
    https://arxiv.org/pdf/1905.11946.pdf
'''


class EfficientNet(nn.Module):
    def __init__(self, subtype='efficientnet_b0', out_stages=[3, 5, 7], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):

        super(EfficientNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        self.depth_div = 8

        if self.subtype == 'efficientnet_b0':
            efficientnet = models.efficientnet_b0(pretrained=self.pretrained)
            self.out_channels = [32, 16, 24, 40, 80, 112, 192, 320]
        elif self.subtype == 'efficientnet_b1':
            efficientnet = models.efficientnet_b1(pretrained=self.pretrained)
            self.out_channels = [32, 16, 24, 40, 80, 112, 192, 320]
        elif self.subtype == 'efficientnet_b2':
            efficientnet = models.efficientnet_b2(pretrained=self.pretrained)
            self.out_channels = [32, 16, 24, 48, 88, 120, 208, 352]
        elif self.subtype == 'efficientnet_b3':
            efficientnet = models.efficientnet_b3(pretrained=self.pretrained)
            self.out_channels = [40, 24, 32, 48, 96, 136, 232, 384]
        elif self.subtype == 'efficientnet_b4':
            efficientnet = models.efficientnet_b4(pretrained=self.pretrained)
            self.out_channels = [48, 24, 32, 56, 112, 160, 272, 448]
        elif self.subtype == 'efficientnet_b5':
            efficientnet = models.efficientnet_b5(pretrained=self.pretrained)
            self.out_channels = [48, 24, 40, 64, 128, 176, 304, 512]
        elif self.subtype == 'efficientnet_b6':
            efficientnet = models.efficientnet_b6(pretrained=self.pretrained)
            self.out_channels = [56, 32, 40, 72, 144, 200, 344, 576]
        elif self.subtype == 'efficientnet_b7':
            efficientnet = models.efficientnet_b7(pretrained=self.pretrained)
            self.out_channels = [64, 32, 48, 80, 160, 224, 384, 640]
        else:
            raise NotImplementedError

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        features = list(efficientnet.features.children())
        self.stem = features[0] # x2
        self.stage1 = features[1]
        self.stage2 = features[2]
        self.stage3 = features[3]
        self.stage4 = features[4]
        self.stage5 = features[5]
        self.stage6 = features[6]
        self.stage7 = features[7]
        if self.classifier:
            self.last_conv = features[8]
            self.avgpool = efficientnet.avgpool
            self.fc = efficientnet.classifier
            self.fc[1] = nn.Linear(self.fc[1].in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 8):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.last_conv(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=='__main__':
    model = EfficientNet('efficientnet_b0', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)