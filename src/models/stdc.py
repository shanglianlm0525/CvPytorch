# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/5 10:18
# @Author : liumin
# @File : stdc.py

"""
    Rethinking BiSeNet For Real-time Semantic Segmentation
    https://arxiv.org/pdf/2104.13188.pdf
"""
import math
from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.losses.seg_loss import CrossEntropyLoss2d, FocalLoss, DetailAggregateLoss, OhemCrossEntropyLoss2d


class STDC(nn.Module):
    def __init__(self, dictionary=None):
        super().__init__()
        self.dictionary = dictionary
        self.input_size = [512, 1024]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        # backbone_cfg = {'name': 'STDCNet', 'subtype': 'stdc1', 'out_stages': [3, 4, 5], 'output_stride': 32,
        #                'pretrained': True, 'backbone_path': './weights/stdc/STDCNet813M_73.91.tar'}
        backbone_cfg = {'name': 'STDCNet', 'subtype': 'stdc2', 'out_stages': [3, 4, 5], 'output_stride': 32,
                        'pretrained': True, 'backbone_path': './weights/stdc/STDCNet1446_76.47.tar'}
        self.backbone = build_backbone(backbone_cfg)

        head_cfg = {'name': 'StdcHead', 'in_channels': self.backbone.out_channels, 'num_classes': self.num_classes}
        self.head = build_head(head_cfg)

        self.ce_criterion = OhemCrossEntropyLoss2d(thresh=0.7).cuda()
        self.boundary_criterion = DetailAggregateLoss().cuda()

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        batch_size, ch, h, w = imgs.shape

        feats = self.backbone(imgs)
        feat_outs = self.head(feats)
        out8 = F.interpolate(feat_outs[0], (h, w), mode='bilinear', align_corners=True)
        out16 = F.interpolate(feat_outs[1], (h, w), mode='bilinear', align_corners=True)
        out32 = F.interpolate(feat_outs[2], (h, w), mode='bilinear', align_corners=True)

        if mode == 'infer':

            return torch.argmax(out8, dim=1)
        else:
            losses = {}

            losses['ce_loss_8'] = self.ce_criterion(out8, targets)
            losses['ce_loss_16'] = self.ce_criterion(out16, targets)
            losses['ce_loss_32'] = self.ce_criterion(out32, targets)

            losses['boundary_bce_loss_8'], losses['boundary_dice_loss_8'] = self.boundary_criterion(feat_outs[3], targets)

            losses['loss'] = losses['ce_loss_8'] + losses['ce_loss_16'] + losses['ce_loss_32'] + \
                             losses['boundary_bce_loss_8'] + losses['boundary_dice_loss_8']

            if mode == 'val':
                return losses, torch.argmax(out8, dim=1)
            else:
                return losses
