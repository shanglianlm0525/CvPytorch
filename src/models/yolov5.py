# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/2 13:03
# @Author : liumin
# @File : yolov5.py

import torch
import torch.nn as nn

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head


class YOLOv5(nn.Module):
    def __init__(self, dictionary=None):
        super(YOLOv5, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        backbone_cfg = {'name': 'YOLOv5Backbone', 'subtype': 'yolov5s', 'out_stages': [2, 3, 4], 'output_stride': 32}
        self.backbone = build_backbone(backbone_cfg)
        neck_cfg = {'name': 'YOLOv5Neck', 'in_channels': self.backbone.out_channels, 'out_channels': self.backbone.out_channels }
        self.neck = build_neck(neck_cfg)
        head_cfg = {'name': 'YOLOv5Neck', 'in_channels': self.backbone.out_channels,
                    'out_channels': self.backbone.out_channels}
        self.head = build_head(head_cfg)
        print(self.backbone)
        print(self.neck)
        print(self.head)



    def forward(self, imgs, targets=None, mode='infer', **kwargs):

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            losses = {}
            x = self.backbone(imgs)
            x = self.fpn(x)
            x = self.head(x)
            preds = tuple(x)

            loss, loss_states = self.head.loss(preds, targets)

            losses['qfl_loss'] = loss_states['loss_qfl']
            losses['bbox_loss'] = loss_states['loss_bbox']
            losses['dfl_loss'] = loss_states['loss_dfl']
            losses['loss'] = loss.mean()

            if mode == 'val':
                dets = self.head.post_process(preds, targets)
                return losses, dets
            else:
                return losses
