# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/28 14:19
# @Author : liumin
# @File : nanodet.py

import warnings
import torch
import torch.nn as nn
import numpy as np

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from ..losses.nanodet_loss import NanoDetLoss


class NanoDet(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(NanoDet, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 320, 320)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)

        # self.model_cfg.LOSS.num_classes = self.num_classes
        # self.loss = NanoDetLoss(self.model_cfg.LOSS)

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def trans_specific_format(self, imgs, targets):
        new_boxes = []
        new_labels = []
        new_scales = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_boxes.append(target['boxes'])
            new_labels.append(target['labels'])
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["boxes"] = new_boxes
        t_targets["labels"] = new_labels
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return torch.stack(imgs), t_targets


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05
        if mode == 'infer':
            pass
        else:
            losses = {}
            imgs, targets = self.trans_specific_format(imgs, targets)

            preds = self.head(self.neck(self.backbone(imgs)))

            loss, loss_states = self.head.loss(preds, targets)

            losses['qfl_loss'] = loss_states["loss_qfl"]
            losses['bbox_loss'] = loss_states["loss_bbox"]
            losses['dfl_loss'] = loss_states["loss_dfl"]
            losses['loss'] = loss

            if mode == 'val':
                outputs = []
                dets = self.head.post_process(preds, imgs)
                for width, height, scale, det in zip(targets['width'], targets['height'], targets['scales'],dets):
                    det_bboxes, det_labels = det
                    det_bboxes_np = det_bboxes[:, :4].cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    scale = scale.cpu().numpy()
                    det_bboxes_np[:, [0, 2]] /= scale[1]
                    det_bboxes_np[:, [1, 3]] /= scale[0]
                    # clip boxes
                    det_bboxes_np[:, [0, 2]] = det_bboxes_np[:, [0, 2]].clip(0, width)
                    det_bboxes_np[:, [1, 3]] = det_bboxes_np[:, [1, 3]].clip(0, height)
                    outputs.append({"boxes": torch.tensor(det_bboxes_np), "labels": det_labels, "scores": det_bboxes[:, 4]})
                return losses, outputs
            else:
                return losses
