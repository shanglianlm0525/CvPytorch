# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/6 16:13
# @Author : liumin
# @File : vgg.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn, vgg11, vgg13, vgg16, vgg19

"""
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/pdf/1409.1556.pdf
"""

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):

    def __init__(self, subtype='vgg16', out_stages=[2,3,4], output_stride=32, classifier=False, num_classes=1000, backbone_path=None, pretrained = False):
        super(VGG, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'vgg11':
            features = vgg11(self.pretrained).features
            classifier = vgg11(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:3])
            self.layer1 = nn.Sequential(*list(features.children())[3:5])
            self.layer1_pool = nn.Sequential(list(features.children())[5])
            self.layer2 = nn.Sequential(*list(features.children())[6:10])
            self.layer2_pool = nn.Sequential(list(features.children())[10])
            self.layer3 = nn.Sequential(*list(features.children())[11:15])
            self.layer3_pool = nn.Sequential(list(features.children())[15])
            self.layer4 = nn.Sequential(*list(features.children())[16:20])
            self.layer4_pool = nn.Sequential(list(features.children())[20])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg13':
            features = vgg13(self.pretrained).features
            classifier = vgg13(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:5])
            self.layer1 = nn.Sequential(*list(features.children())[5:9])
            self.layer1_pool = nn.Sequential(list(features.children())[9])
            self.layer2 = nn.Sequential(*list(features.children())[10:14])
            self.layer2_pool = nn.Sequential(list(features.children())[14])
            self.layer3 = nn.Sequential(*list(features.children())[15:19])
            self.layer3_pool = nn.Sequential(list(features.children())[19])
            self.layer4 = nn.Sequential(*list(features.children())[20:24])
            self.layer4_pool = nn.Sequential(list(features.children())[24])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg16':
            features = vgg16(self.pretrained).features
            classifier = vgg16(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:5])
            self.layer1 = nn.Sequential(*list(features.children())[5:9])
            self.layer1_pool = nn.Sequential(list(features.children())[9])
            self.layer2 = nn.Sequential(*list(features.children())[10:16])
            self.layer2_pool = nn.Sequential(list(features.children())[16])
            self.layer3 = nn.Sequential(*list(features.children())[17:23])
            self.layer3_pool = nn.Sequential(list(features.children())[23])
            self.layer4 = nn.Sequential(*list(features.children())[24:30])
            self.layer4_pool = nn.Sequential(list(features.children())[30])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg19':
            features = vgg19(self.pretrained).features
            classifier = vgg19(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:5])
            self.layer1 = nn.Sequential(*list(features.children())[5:9])
            self.layer1_pool = nn.Sequential(list(features.children())[9])
            self.layer2 = nn.Sequential(*list(features.children())[10:18])
            self.layer2_pool = nn.Sequential(list(features.children())[18])
            self.layer3 = nn.Sequential(*list(features.children())[19:27])
            self.layer3_pool = nn.Sequential(list(features.children())[27])
            self.layer4 = nn.Sequential(*list(features.children())[28:36])
            self.layer4_pool = nn.Sequential(list(features.children())[36])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg11_bn':
            features = vgg11_bn(self.pretrained).features
            classifier = vgg11_bn(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:4])
            self.layer1 = nn.Sequential(*list(features.children())[4:7])
            self.layer1_pool = nn.Sequential(list(features.children())[7])
            self.layer2 = nn.Sequential(*list(features.children())[8:14])
            self.layer2_pool = nn.Sequential(list(features.children())[14])
            self.layer3 = nn.Sequential(*list(features.children())[15:21])
            self.layer3_pool = nn.Sequential(list(features.children())[21])
            self.layer4 = nn.Sequential(*list(features.children())[22:28])
            self.layer4_pool = nn.Sequential(list(features.children())[28])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg13_bn':
            features = vgg13_bn(self.pretrained).features
            classifier = vgg13_bn(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:7])
            self.layer1 = nn.Sequential(*list(features.children())[7:13])
            self.layer1_pool = nn.Sequential(list(features.children())[13])
            self.layer2 = nn.Sequential(*list(features.children())[14:20])
            self.layer2_pool = nn.Sequential(list(features.children())[20])
            self.layer3 = nn.Sequential(*list(features.children())[20:27])
            self.layer3_pool = nn.Sequential(list(features.children())[27])
            self.layer4 = nn.Sequential(*list(features.children())[28:34])
            self.layer4_pool = nn.Sequential(list(features.children())[34])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg16_bn':
            features = vgg16_bn(self.pretrained).features
            classifier = vgg16_bn(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:7])
            self.layer1 = nn.Sequential(*list(features.children())[7:13])
            self.layer1_pool = nn.Sequential(list(features.children())[13])
            self.layer2 = nn.Sequential(*list(features.children())[14:23])
            self.layer2_pool = nn.Sequential(list(features.children())[23])
            self.layer3 = nn.Sequential(*list(features.children())[24:33])
            self.layer3_pool = nn.Sequential(list(features.children())[33])
            self.layer4 = nn.Sequential(*list(features.children())[34:43])
            self.layer4_pool = nn.Sequential(list(features.children())[43])

            self.out_channels = [64, 128, 256, 512, 512]
        elif self.subtype == 'vgg19_bn':
            features = vgg19_bn(self.pretrained).features
            classifier = vgg19_bn(self.pretrained).classifier
            self.conv1 = nn.Sequential(*list(features.children())[:7])
            self.layer1 = nn.Sequential(*list(features.children())[7:13])
            self.layer1_pool = nn.Sequential(list(features.children())[13])
            self.layer2 = nn.Sequential(*list(features.children())[14:26])
            self.layer2_pool = nn.Sequential(list(features.children())[26])
            self.layer3 = nn.Sequential(*list(features.children())[27:39])
            self.layer3_pool = nn.Sequential(list(features.children())[39])
            self.layer4 = nn.Sequential(*list(features.children())[40:52])
            self.layer4_pool = nn.Sequential(list(features.children())[52])

            self.out_channels = [64, 128, 256, 512, 512]
        else:
            raise NotImplementedError

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        if self.classifier:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.fc = classifier
            self.fc[-1] = nn.Linear(classifier[-1].in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained and self.backbone_path is not None:
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
        x = self.conv1(x)
        output = []
        for i in range(1, min(max(self.out_stages)+1, 5)):
            layer = getattr(self, 'layer{}'.format(i))
            x = layer(x)
            if i in self.out_stages:
                output.append(x)
            layer_pool = getattr(self, 'layer{}_pool'.format(i))
            x = layer_pool(x)

        if self.classifier:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
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
    model =VGG('vgg19', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)