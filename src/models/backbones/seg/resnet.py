# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/3 9:02
# @Author : liumin
# @File : resnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from src.models.bricks import build_conv_layer, build_norm_layer

"""
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
"""


model_urls = {
    "resnet18v1c": "https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth",
    "resnet50v1c": "https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth",
    "resnet101v1c": "https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth",
}

class ResNet(nn.Module):

    def __init__(self, subtype='resnet50', out_stages=[2, 3, 4], output_stride=32, frozen_stages=-1, norm_eval=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True),
                 classifier=False, num_classes=1000, backbone_path=None, pretrained = True):
        super(ResNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.subtype in ['resnet18', 'resnet18v1c', 'resnet18v1d']:
            resnet = resnet18(self.pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif self.subtype in ['resnet34', 'resnet34v1c', 'resnet34v1d']:
            resnet = resnet34(self.pretrained)
            self.out_channels = [64, 64, 128, 256, 512]
        elif self.subtype in ['resnet50', 'resnet50v1c', 'resnet50v1d']:
            resnet = resnet50(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif self.subtype in ['resnet101', 'resnet101v1c', 'resnet101v1d']:
            resnet = resnet101(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        elif self.subtype in ['resnet152', 'resnet152v1c', 'resnet152v1d']:
            resnet = resnet152(self.pretrained)
            self.out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.deep_stem = True if self.subtype.endswith('c') or self.subtype.endswith('d') else False
        self.avg_down = True if self.subtype.endswith('d') else False

        if self.deep_stem:
            mid_channels = self.out_channels[0] // 2
            self.stem = nn.Sequential(
                build_conv_layer(self.conv_cfg, 3, mid_channels, kernel_size=3, stride=2, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, mid_channels)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, mid_channels, mid_channels, kernel_size=3,stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, mid_channels)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(self.conv_cfg, mid_channels, self.out_channels[0],
                    kernel_size=3, stride=1, padding=1, bias=False),
                build_norm_layer(self.norm_cfg, self.out_channels[0])[1],
                nn.ReLU(inplace=True))
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        else:
            self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
            self.maxpool = resnet.maxpool

        if self.avg_down:
            self.layer1 = self.trans_avg_down(resnet.layer1)
            self.layer2 = self.trans_avg_down(resnet.layer2)
            self.layer3 = self.trans_avg_down(resnet.layer3)
            self.layer4 = self.trans_avg_down(resnet.layer4)
        else:
            self.layer1 = resnet.layer1
            self.layer2 = resnet.layer2
            self.layer3 = resnet.layer3
            self.layer4 = resnet.layer4

        if self.classifier:
            self.avgpool = resnet.avgpool
            self.fc = nn.Linear(resnet.fc.in_features, self.num_classes)
            self.out_channels = self.num_classes


        if self.output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif self.output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

            for n, m in self.layer3.named_modules():
                if 'conv1' in n and 'resnet34' in subtype:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n and 'resnet18' in subtype:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)

        if self.output_stride == 8 or self.output_stride == 16:
            for n, m in self.layer4.named_modules():
                if 'conv1' in n and 'resnet34' in subtype:
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
                elif 'conv2' in n and 'resnet18' in subtype:
                    m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        if (self.pretrained and self.backbone_path is not None) or\
                self.subtype in ["resnet18v1c", "resnet50v1c", "resnet101v1c"]:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def trans_avg_down(self, layer):
        if layer[0].downsample is not None:
            stride = layer[0].downsample[0].stride[0]
            layer[0].downsample[0].stride = (1, 1)
            layer[0].downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
                layer[0].downsample[0],
                layer[0].downsample[1]
            )
        return layer

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)

        if self.classifier:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.stem[1].eval()
                for m in [self.stem[0], self.stem[1]]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))
        elif self.subtype in ["resnet18v1c", "resnet50v1c", "resnet101v1c"]:
            print('=> loading pretrained model {}'.format(model_urls[self.subtype]))
            state_dict = load_state_dict_from_url(model_urls[self.subtype], progress=True)['state_dict']
            model_state_dict = self.state_dict()
            for (k1, v1), (k2, v2) in zip(model_state_dict.items(), state_dict.items()):
                if v1.shape == v2.shape:
                    model_state_dict.update({k1: v2})
            self.load_state_dict(model_state_dict)


if __name__ == "__main__":
    model = ResNet('resnet18v1c', output_stride=8)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)