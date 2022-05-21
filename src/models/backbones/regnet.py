# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/2/17 19:05
# @Author : liumin
# @File : regnet.py


import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models as models
from torchvision.models.regnet import model_urls

"""
    Designing Network Design Spaces
    https://arxiv.org/pdf/2003.13678.pdf
"""

class RegNet(nn.Module):
    def __init__(self, subtype='regnet_y_400mf', out_stages=[2, 3, 4], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):
        super(RegNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if self.subtype == 'regnet_y_400mf':
            regnet = models.regnet_y_400mf(pretrained=self.pretrained)
            self.out_channels = [32, 48, 104, 208, 440]
        elif self.subtype == 'regnet_y_800mf':
            regnet = models.regnet_y_800mf(pretrained=self.pretrained)
            self.out_channels = [32, 64, 144, 320, 784]
        elif self.subtype == 'regnet_y_1_6gf':
            regnet = models.regnet_y_1_6gf(pretrained=self.pretrained)
            self.out_channels = [32, 48, 120, 336, 888]
        elif self.subtype == 'regnet_y_3_2gf':
            regnet = models.regnet_y_3_2gf(pretrained=self.pretrained)
            self.out_channels = [32, 72, 216, 576, 1512]
        elif self.subtype == 'regnet_y_8gf':
            regnet = models.regnet_y_8gf(pretrained=self.pretrained)
            self.out_channels = [32, 224, 448, 896, 2016]
        elif self.subtype == 'regnet_y_16gf':
            regnet = models.regnet_y_16gf(pretrained=self.pretrained)
            self.out_channels = [32, 224, 448, 1232, 3024]
        elif self.subtype == 'regnet_y_32gf':
            regnet = models.regnet_y_32gf(pretrained=self.pretrained)
            self.out_channels = [32, 232, 696, 1392, 3712]
        elif self.subtype == 'regnet_y_128gf':
            regnet = models.regnet_y_128gf(pretrained=self.pretrained)
            self.out_channels = [32, 528, 1056, 2904, 7392]
        elif self.subtype == 'regnet_x_400mf':
            regnet = models.regnet_x_400mf(pretrained=self.pretrained)
            self.out_channels = [32, 32, 64, 160, 400]
        elif self.subtype == 'regnet_x_800mf':
            regnet = models.regnet_x_800mf(pretrained=self.pretrained)
            self.out_channels = [32, 64, 128, 288, 672]
        elif self.subtype == 'regnet_x_1_6gf':
            regnet = models.regnet_x_1_6gf(pretrained=self.pretrained)
            self.out_channels = [32, 72, 168, 408, 912]
        elif self.subtype == 'regnet_x_3_2gf':
            regnet = models.regnet_x_3_2gf(pretrained=self.pretrained)
            self.out_channels = [32, 96, 192, 432, 1008]
        elif self.subtype == 'regnet_x_8gf':
            regnet = models.regnet_x_8gf(pretrained=self.pretrained)
            self.out_channels = [32, 80, 240, 720, 1920]
        elif self.subtype == 'regnet_x_16gf':
            regnet = models.regnet_x_16gf(pretrained=self.pretrained)
            self.out_channels = [32, 256, 512, 896, 2048]
        elif self.subtype == 'regnet_x_32gf':
            regnet = models.regnet_x_32gf(pretrained=self.pretrained)
            self.out_channels = [32, 336, 672, 1344, 2520]
        else:
            raise NotImplementedError

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        self.stem = regnet.stem # x2
        self.stage1 = regnet.trunk_output.block1
        self.stage2 = regnet.trunk_output.block2
        self.stage3 = regnet.trunk_output.block3
        self.stage4 = regnet.trunk_output.block4
        if self.classifier:
            self.avgpool = regnet.avgpool
            self.fc = nn.Linear(regnet.fc.in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.last_conv(x)
            x = self.avgpool(x)
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

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    model = RegNet('regnet_y_400mf')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
