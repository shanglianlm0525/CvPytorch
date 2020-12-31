# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 18:36
# @Author : liumin
# @File : shufflenet_v2.py

import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0

"""
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/abs/1807.11164
"""

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
}

class ShuffleNetV2(nn.Module):

    def __init__(self, subtype='shufflenet_v2_x1_0', out_stages=[2,3,4], backbone_path=None):
        super(ShuffleNetV2, self).__init__()
        self.out_stages = out_stages
        self.backbone_path = backbone_path

        if subtype == 'shufflenet_v2_x0_5':
            self.backbone = shufflenet_v2_x0_5(pretrained=not self.backbone_path)
            self.out_channels = [24, 48, 96, 192, 1024]
        elif subtype == 'shufflenet_v2_x1_0':
            self.backbone = shufflenet_v2_x1_0(pretrained=not self.backbone_path)
            self.out_channels = [24, 116, 232, 464, 1024]
        elif subtype == 'shufflenet_v2_x1_5':
            self.backbone = shufflenet_v2_x1_5(pretrained=not self.backbone_path)
            self.out_channels = [24, 176, 352, 704, 1024]
        elif subtype == 'shufflenet_v2_x2_0':
            self.backbone = shufflenet_v2_x2_0(pretrained=not self.backbone_path)
            self.out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        self.init_weights()

        if self.backbone_path:
            self.backbone.load_state_dict(torch.load(self.backbone_path))


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self.backbone, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)


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


if __name__=="__main__":
    model =ShuffleNetV2('shufflenet_v2_x1_0')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)