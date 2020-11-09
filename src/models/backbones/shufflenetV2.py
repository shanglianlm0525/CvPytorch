# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:36
# @Author : liumin
# @File : shufflenetV2.py

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0


class ShuffleNetV2(nn.Module):
    '''
        ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
        https://arxiv.org/abs/1807.11164
    '''
    def __init__(self, backbone='shufflenet_v2_x1_0', backbone_path=None, use_fpn=True):
        super(ShuffleNetV2, self).__init__()
        self.use_fpn = use_fpn

        if backbone == 'shufflenet_v2_x0_5':
            backbone = shufflenet_v2_x0_5(pretrained=not backbone_path)
            self.out_channels = [48, 96, 192]
        elif backbone == 'shufflenet_v2_x1_0':
            backbone = shufflenet_v2_x1_0(pretrained=not backbone_path)
            self.out_channels = [116, 232, 464]
        elif backbone == 'shufflenet_v2_x1_5':
            backbone = shufflenet_v2_x1_5(pretrained=not backbone_path)
            self.out_channels = [176, 352, 704]
        elif backbone == 'shufflenet_v2_x2_0':
            backbone = shufflenet_v2_x2_0(pretrained=not backbone_path)
            self.out_channels = [244, 488, 976]
        else:
            raise NotImplementedError

        if backbone_path:
            backbone.load_state_dict(torch.load(backbone_path))

        self.conv1 = backbone.conv1
        self.maxpool = backbone.maxpool
        self.layer2 = backbone.stage2
        self.layer3 = backbone.stage3
        self.layer4 = backbone.stage4

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        out3 = self.layer2(x)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        if self.use_fpn:
            return out3, out4, out5
        else:
            return out5

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=="__main__":
    model =ShuffleNetV2('shufflenet_v2_x1_0')
    print(model)