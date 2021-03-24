# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/20 13:19
# @Author : liumin
# @File : darknet.py

import torch
import torch.nn as nn
from CvPytorch.src.models.modules.convs import ConvModule

"""
    YOLOv3: An Incremental Improvement
    https://pjreddie.com/media/files/papers/YOLOv3.pdf
"""


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        assert channels % 2 == 0  # ensure the in_channels is even
        mid_channels = channels // 2

        self.conv1 = ConvModule(channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False,
                                norm='BatchNorm2d', activation='ReLU')
        self.conv2 = ConvModule(mid_channels, channels, kernel_size=3, stride=1, padding=1, bias=False,
                                norm='BatchNorm2d', activation='ReLU')

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return out


class Darknet(nn.Module):
    def __init__(self, subtype='darknet53', out_stages=[2, 3, 4], backbone_path=None, pretrained = False):
        super(Darknet, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        self.out_channels = [64, 128, 256, 512, 1024]
        self.conv1 = ConvModule(3, 32, kernel_size=3, stride=1, padding=1, bias=False,
                   norm='BatchNorm2d', activation='ReLU')

        if subtype == 'darknet53':
            self.layer1 = self._make_layer_v3(32, 64, 1, 2)
            self.layer2 = self._make_layer_v3(64, 128, 2, 2)
            self.layer3 = self._make_layer_v3(128, 256, 8, 2)
            self.layer4 = self._make_layer_v3(256, 512, 8, 2)
            self.layer5 = self._make_layer_v3(512, 1024, 4, 2)
        elif subtype == 'darknet19':
            self.layer1 = self._make_layer_v2(32, 64, 1, 2)
            self.layer2 = self._make_layer_v2(64, 128, 3, 2)
            self.layer3 = self._make_layer_v2(128, 256, 3, 2)
            self.layer4 = self._make_layer_v2(256, 512, 5, 2)
            self.layer5 = self._make_layer_v2(512, 1024, 5, 2)
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        if not self.pretrained:
            if self.backbone_path:
                self.pretrained = True
                self.backbone.load_state_dict(torch.load(self.backbone_path))
            else:
                self.init_weights()

    def _make_layer_v2(self, in_places, places, block, stride):
        layers = []
        layers.append(nn.MaxPool2d(kernel_size=2, stride=stride))
        for i in range(block):
            layers.append(ConvModule(in_places, places, kernel_size=1 if i%2 else 3, stride=1, padding=(i+1)%2, bias=False,
                            norm='BatchNorm2d', activation='ReLU'))
            in_places, places = places, in_places
        return nn.Sequential(*layers)

    def _make_layer_v3(self, in_places, places, block, stride):
        layers = []
        layers.append(ConvModule(in_places, places, kernel_size=3, stride=stride, padding=1, bias=False,
                                 norm='BatchNorm2d', activation='ReLU'))
        for i in range(block):
            layers.append(ResBlock(places))
        return nn.Sequential(*layers)


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
        for i in range(1, 6):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)

        return tuple(output)




if __name__ == "__main__":
    model = Darknet(subtype='darknet19')
    inputs = torch.rand(1, 3, 416, 416)
    level_outputs = model(inputs)
    for level_out in level_outputs:
        print(tuple(level_out.shape))