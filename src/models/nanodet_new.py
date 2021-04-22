# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:46
# @Author : liumin
# @File : nanodet_new.py

import torch
import torch.nn as nn

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from .heads.nanodet_head import NanodetLoss


class NanoDet(nn.Module):
    def __init__(self, dictionary=None):
        super(NanoDet, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self.num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        backbone_cfg = {'name': 'ShuffleNetV2', 'subtype':'shufflenet_v2_x1_0', 'out_stages': [2, 3, 4]}
        neck_cfg = {'name': 'PAN', 'in_channels': [116, 232, 464], 'out_channels': 96}
        head_cfg = {'name': 'NanodetHead', 'num_classes': 80, 'input_channels': [96, 96, 96], 'feat_channels': 96,
                    'stacked_convs': 2, 'reg_max': 7,'share_cls_reg': True, 'use_sigmoid': True}

        loss_cfg = {'num_classes': 80, 'strides': [8, 16, 32],'reg_max': 7,'use_sigmoid': True}
        '''
        loss_cfg = {'loss_qfl': {'name': 'QualityFocalLoss', 'use_sigmoid': True, 'beta': 2.0, 'loss_weight': 1.0},
         'loss_dfl': {'name': 'DistributionFocalLoss', 'loss_weight': 0.25},
         'loss_bbox': {'name': 'GIoULoss', 'loss_weight': 2.0}
         }
        '''
        self.backbone = build_backbone(backbone_cfg) # ShuffleNetV2(**backbone_cfg)
        self.neck = build_neck(neck_cfg) # PAN(**fpn_cfg)
        self.head = build_head(head_cfg) # NanoDetHead(**head_cfg)
        self.loss = NanodetLoss(**loss_cfg)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            losses = {}
            x = self.backbone(imgs)
            x = self.neck(x)
            x = self.head(x)
            preds = tuple(x)

            loss, loss_states = self.loss(preds, targets)

            losses['qfl_loss'] = loss_states['loss_qfl']
            losses['bbox_loss'] = loss_states['loss_bbox']
            losses['dfl_loss'] = loss_states['loss_dfl']
            losses['loss'] = loss.mean()

            if mode == 'val':
                dets = self.head.post_process(preds, targets)
                return losses, dets
            else:
                return losses