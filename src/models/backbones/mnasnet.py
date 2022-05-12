# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/1 9:38
# @Author : liumin
# @File : mnasnet.py


import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.mnasnet import mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
from torchvision.models.mnasnet import _MODEL_URLS as model_urls

"""
    MnasNet: Platform-Aware Neural Architecture Search for Mobile
    https://arxiv.org/pdf/1807.11626.pdf
"""


class MNASNet(nn.Module):

    def __init__(self, subtype='mnasnet0_5', out_stages=[3, 5, 7], output_stride=16, classifier=False, num_classes=1000, pretrained = False, backbone_path=None):
        super(MNASNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path


        if self.subtype == 'mnasnet0_5':
            mnasnet = mnasnet0_5(self.pretrained)
            self.out_channels = [16, 8, 12, 20, 40, 48, 96, 160]
        elif self.subtype == 'mnasnet0_75':
            mnasnet = mnasnet0_75(self.pretrained)
            self.out_channels = [24, 12, 18, 30, 60, 72, 144, 240]
        elif self.subtype == 'mnasnet1_0':
            mnasnet = mnasnet1_0(self.pretrained)
            self.out_channels = [32, 16, 24, 40, 80, 96, 192, 320]
        elif self.subtype == 'mnasnet1_3':
            mnasnet = mnasnet1_3(self.pretrained)
            self.out_channels = [40, 24, 32, 56, 104, 128, 248, 416]
        else:
            raise NotImplementedError


        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        layers = list(mnasnet.layers.children())

        self.stem = nn.Sequential(*layers[0:3]) # x2
        self.stage1 = nn.Sequential(*layers[3:8])
        self.stage2 = layers[8]
        self.stage3 = layers[9]
        self.stage4 = layers[10]
        self.stage5 = layers[11]
        self.stage6 = layers[12]
        self.stage7 = layers[13]

        if self.classifier:
            self.last_conv = nn.Sequential(*layers[14:])
            self.fc = mnasnet.classifier
            self.fc[1] = nn.Linear(self.fc[1].in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained and self.backbone_path is not None:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)


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
            # Equivalent to global avgpool and removing H and W dimensions.
            x = x.mean([2, 3])
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]


    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__=="__main__":
    model = MNASNet('mnasnet0_5')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)