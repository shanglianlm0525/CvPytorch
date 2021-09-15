# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/1 13:16
# @Author : liumin
# @File : yolop.py

import torch
import torch.nn as nn

from src.losses.yolop_loss import YolopLoss
from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.models.necks import build_neck


class YOLOP(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOP, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.conf_thr = 0.01
        self.nms_thr = 0.65

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)

        self.loss = YolopLoss(self.num_classes)

        self._init_weight()

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)
            b, _, height, width = imgs.shape

            losses = {}
            x = self.backbone(imgs)
            x = self.neck(x)
            det_out, det_train_out, seg_drivable_area, seg_lane_line = self.head(x)
            '''
                det_out: [torch.Size([1, 3, 80, 80, 6]), torch.Size([1, 3, 40, 40, 6]), torch.Size([1, 3, 20, 20, 6])]
                det_train_out: 
                seg_drivable_area: torch.Size([1, 2, 640, 640])
                seg_lane_line: torch.Size([1, 2, 640, 640])
            '''

            if mode == 'val':
                outputs = []

                return losses, outputs
            else:
                return losses
