# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/2 19:31
# @Author : liumin
# @File : pp_liteseg.py

"""
    PP-LiteSeg: A Superior Real-Time Semantic Segmentation Model
    https://arxiv.org/pdf/2204.02681.pdf
"""

from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.losses.seg_loss import CrossEntropyLoss2d, FocalLoss, OhemCrossEntropyLoss2d
from src.utils.torch_utils import set_bn_momentum


class PPLiteSeg(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super().__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.input_size = [512, 1024]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.head = build_head(self.model_cfg.HEAD)

        self.criterion = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()
        # self.criterion = OhemCrossEntropyLoss2d().cuda()

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)

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
        imgs_hw = imgs.shape[2:]
        feats = self.backbone(imgs)
        logits = self.head(feats)
        outputs = [F.interpolate(x, imgs_hw, mode='bilinear', align_corners=False) for x in logits]

        if mode == 'infer':

            return torch.argmax(outputs[0], dim=1)
        else:
            losses = {}
            losses['ce_loss1'] = self.criterion(outputs[0], targets)
            losses['ce_loss2'] = self.criterion(outputs[1], targets)
            losses['ce_loss3'] = self.criterion(outputs[2], targets)
            losses['loss'] = losses['ce_loss1'] + losses['ce_loss2'] + losses['ce_loss3']

            if mode == 'val':
                return losses, torch.argmax(outputs[0], dim=1)
            else:
                return losses