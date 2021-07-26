# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:03
# @Author : liumin
# @File : pspnet.py

"""
    Pyramid Scene Parsing Network
    https://arxiv.org/pdf/1612.01105.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
import numpy as np
from src.losses.seg_loss import BCEWithLogitsLoss2d, CrossEntropyLoss2d
from src.models.backbones import build_backbone

class PPM(nn.Module):
    def __init__(self, channels = 2048):
        super(PPM, self).__init__()
        bins = (1, 2, 3, 6)
        reduction_dim = int(channels / len(bins))
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(channels, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class PSPNet(nn.Module):
    def __init__(self, dictionary=None):
        super(PSPNet, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        backbone_cfg = {'name': 'ResNet', 'subtype': 'resnet50', 'out_stages': [3, 4], 'output_stride':8, 'pretrained': True}
        self.backbone = build_backbone(backbone_cfg)
        aux_b_channels, b_channels = self.backbone.out_channels[0], self.backbone.out_channels[1]
        self.ppm = PPM(b_channels)
        self.cls = nn.Sequential(
            nn.Conv2d(b_channels*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1)
        )

        self.aux = nn.Sequential(
            nn.Conv2d(aux_b_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, self.num_classes, kernel_size=1)
        )

        self._init_weight(self.cls, self.aux)
        # self.bce_criterion = BCEWithLogitsLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()
        self.criterion = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()

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
        aux_x, x  = self.backbone(imgs)
        x = self.ppm(x)
        x = self.cls(x)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)


        if mode == 'infer':

            return torch.argmax(outputs, dim=1)
        else:
            losses = {}
            losses['loss'] = 0
            losses['ce_loss'] = self.criterion(outputs, targets)
            losses['loss'] = losses['ce_loss']

            if mode == 'val':
                return losses, torch.argmax(outputs, dim=1)
            else:
                aux = self.aux(aux_x)
                aux = F.interpolate(aux, size=imgs.size()[2:], mode='bilinear', align_corners=True)

                aux_weight = 0.4
                losses['aux_loss'] = self.criterion(aux, targets)
                losses['loss'] = losses['ce_loss'] + losses['aux_loss'] * aux_weight
                return losses