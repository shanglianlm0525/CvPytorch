# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:58
# @Author : liumin
# @File : squeezenet.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1, model_urls

"""
    SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    https://arxiv.org/abs/1602.07360
"""


class SqueezeNet(nn.Module):

    def __init__(self, subtype='shufflenetv2_x1.0', out_stages=[1, 2, 3], output_stride=32, classifier=False, num_classes=1000, pretrained=True, backbone_path=None):
        super(SqueezeNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if self.subtype == 'squeezenet1_1':
            squeezenet = squeezenet1_1(pretrained=self.pretrained)
            self.out_channels = [96, 128, 256, 512]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        features = list(squeezenet.features.children())
        self.stem = nn.Sequential(*features[0:2])
        self.layer1 = nn.Sequential(*features[2:5])
        self.layer2 = nn.Sequential(*features[5:8])
        self.layer3 = nn.Sequential(*features[8:])

        if self.classifier:
            self.fc = squeezenet.classifier
            self.fc[1] = nn.Linear(self.fc[1].in_features, self.num_classes)
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
        for i in range(1, 4):
            stage = getattr(self, 'layer{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
            if self.classifier:
                x = self.fc(x)
                return torch.flatten(x, 1)

        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

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
    model = SqueezeNet('squeezenet1_1')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
