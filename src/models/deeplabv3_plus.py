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
from src.losses.seg_loss import BCEWithLogitsLoss2d

class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
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

        backbone_cfg = {'name': 'ResNet', 'subtype': 'resnet50', 'out_stages': [3, 4], 'output_stride':8}
        self.backbone = build_backbone(backbone_cfg)
        self.aspp = ASPP(inplanes=self.backbone.out_channels[-1])
        self.decoder = Decoder(self.num_classes, self.backbone.out_channels[0])

        # self._init_weight(self.aspp, self.decoder)

        self.bce_criterion = BCEWithLogitsLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()


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
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)

        if mode == 'infer':

            return outputs
        else:
            losses = {}
            losses['bce_loss'] = self.bce_criterion(outputs, targets)
            losses['loss'] = losses['bce_loss']

            if mode == 'val':
                return losses, torch.argmax(outputs, dim=1).unsqueeze(1)
            else:
                return losses

