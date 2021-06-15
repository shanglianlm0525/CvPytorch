# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:02
# @Author : liumin
# @File : deeplabv3_plus.py

"""
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
    https://arxiv.org/pdf/1802.02611.pdf
"""
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.models.modules.aspp import ASPP
from src.models.backbones import build_backbone
from src.losses.seg_loss import CrossEntropyLoss2d
from src.utils.torch_utils import set_bn_momentum


class Decoder(nn.Module):
    def __init__(self, low_level_channels, in_channels, dilations, num_classes):
        super(Decoder, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(low_level_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(inplace=True))

        self.aspp = ASPP(inplanes=in_channels, dilations = dilations)

        self.classifier = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, high_level_feat, low_level_feat):
        low_level_feat = self.project(low_level_feat)
        high_level_feat = self.aspp(high_level_feat)
        high_level_feat = F.interpolate(high_level_feat, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        return self.classifier(torch.cat((high_level_feat, low_level_feat), dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Deeplabv3Plus(nn.Module):
    def __init__(self, dictionary=None):
        super().__init__()
        self.dictionary = dictionary
        self.input_size = [512, 512]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        backbone_cfg = {'name': 'MobileNetV2', 'subtype': 'mobilenet_v2', 'out_stages': [2, 7], 'output_stride': 16, 'pretrained': True}
        self.backbone = build_backbone(backbone_cfg)
        decoder_cfg = { 'low_level_channels': self.backbone.out_channels[0], 'in_channels': self.backbone.out_channels[1],
                        'dilations': [6, 12, 18], 'num_classes': self.num_classes }
        self.decoder = Decoder(**decoder_cfg)

        self.ce_criterion = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()
        set_bn_momentum(self.backbone, momentum=0.01)


    def _init_weight(self, *stages):
        for m in chain(*stages):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        batch_size, ch, _, _ = imgs.shape
        low_level_feat, x  = self.backbone(imgs)
        x = self.decoder(x, low_level_feat)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)

        if mode == 'infer':

            return outputs
        else:
            losses = {}
            losses['ce_loss'] = self.ce_criterion(outputs, targets)
            losses['loss'] = losses['ce_loss']

            if mode == 'val':
                return losses, outputs.detach().max(dim=1)[1] #torch.argmax(outputs, dim=1)
            else:
                return losses

