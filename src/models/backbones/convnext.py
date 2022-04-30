# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/30 13:32
# @Author : liumin
# @File : convnext.py

import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models as models
from torchvision.models.convnext import _MODELS_URLS as model_urls

'''
    A ConvNet for the 2020s
    https://arxiv.org/pdf/2201.03545.pdf
'''


class ConvNeXt(nn.Module):
    def __init__(self, subtype='convnext_tiny', out_stages=[2, 3, 4], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):

        super(ConvNeXt, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if self.subtype == 'convnext_tiny':
            convnext = models.convnext_tiny(pretrained=self.pretrained)
            self.out_channels = [96, 96, 192, 384, 768]
        elif self.subtype == 'convnext_small':
            convnext = models.convnext_small(pretrained=self.pretrained)
            self.out_channels = [96, 96, 192, 384, 768]
        elif self.subtype == 'convnext_base':
            convnext = models.convnext_base(pretrained=self.pretrained)
            self.out_channels = [128, 128, 256, 512, 1024]
        elif self.subtype == 'convnext_large':
            convnext = models.convnext_large(pretrained=self.pretrained)
            self.out_channels = [256, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        features = list(convnext.features.children())
        self.stem = features[0] # x2
        self.stage1 = features[1]
        self.stage2 = nn.Sequential(*features[2:4])
        self.stage3 = nn.Sequential(*features[4:6])
        self.stage4 = nn.Sequential(*features[6:])
        if self.classifier:
            self.avgpool = convnext.avgpool
            self.fc = convnext.classifier
            self.fc[2] = nn.Linear(self.fc[2].in_features, self.num_classes)
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
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.avgpool(x)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=='__main__':
    model = ConvNeXt('convnext_tiny', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    if isinstance(out, list):
        for o in out:
            print(o.shape)
    else:
        print(out.shape)