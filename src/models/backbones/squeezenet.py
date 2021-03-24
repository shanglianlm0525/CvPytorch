# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:58
# @Author : liumin
# @File : squeezenet.py

import torch
import torch.nn as nn
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1

"""
    SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    https://arxiv.org/abs/1602.07360
"""

model_urls = {
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}

class SqueezeNet(nn.Module):

    def __init__(self, subtype='squeezenet1_1', out_stages=[1, 2, 3], backbone_path=None, pretrained = False):
        super(SqueezeNet, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        self.pretrained = False
        if subtype == 'squeezenet1_1':
            self.pretrained = True
            features = squeezenet1_1(pretrained=self.pretrained).features
            self.out_channels = [96, 128, 256, 512]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.conv1 = nn.Sequential(*list(features.children())[0:2])
        self.layer1 = nn.Sequential(*list(features.children())[2:5])
        self.layer2 = nn.Sequential(*list(features.children())[5:8])
        self.layer3 = nn.Sequential(*list(features.children())[8:13])

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
        x = self.conv1(x)
        output = []
        for i in range(1, 4):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model = SqueezeNet('squeezenet1_1')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
