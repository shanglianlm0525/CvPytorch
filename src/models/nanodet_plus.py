# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/12/27 10:40
# @Author : liumin
# @File : nanodet_plus.py
import copy
import warnings
import torch
import torch.nn as nn
import numpy as np

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head


class NanoDetPlus(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(NanoDetPlus, self).__init__()
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

        # aux
        self.aux_neck = copy.deepcopy(self.neck)
        self.aux_head = build_head(self.model_cfg.AUX_HEAD)

        # self.model_cfg.LOSS.num_classes = self.num_classes
        # self.loss = NanoDetLoss(self.model_cfg.LOSS)

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)
        self.model_cfg.AUX_HEAD.__setitem__('num_classes', self.num_classes)

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
        new_pads = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_boxes.append(target['boxes'])
            new_labels.append(target['labels'])
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["boxes"] = new_boxes
        t_targets["labels"] = new_labels
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05
        if mode == 'infer':
            pass
        else:
            losses = {}
            imgs, targets = self.trans_specific_format(imgs, targets)

            feat = self.backbone(imgs)
            neck_feat = self.neck(feat)
            preds = self.head(neck_feat)

            use_aux = True
            if use_aux:
                aux_neck_feat = self.aux_neck([f.detach() for f in feat])
                dual_fpn_feat = (torch.cat([f, aux_f], dim=1) for f, aux_f in zip(neck_feat, aux_neck_feat))
                aux_preds = self.aux_head(dual_fpn_feat)
                loss, loss_states = self.head.loss(imgs, preds, targets, aux_preds=aux_preds)
            else:
                loss, loss_states = self.head.loss(imgs, preds, targets)

            losses['qfl_loss'] = loss_states["loss_qfl"]
            losses['bbox_loss'] = loss_states["loss_bbox"]
            losses['dfl_loss'] = loss_states["loss_dfl"]

            if use_aux:
                losses['aux_qfl_loss'] = loss_states["aux_loss_qfl"]
                losses['aux_bbox_loss'] = loss_states["aux_loss_bbox"]
                losses['aux_dfl_loss'] = loss_states["aux_loss_dfl"]

            losses['loss'] = loss

            if mode == 'val':
                outputs = []
                dets = self.head.post_process(preds, imgs)
                for width, height, scale, pad, det in zip(targets['width'], targets['height'], targets['scales'], targets['pads'],dets):
                    bboxes, labels = det
                    bboxes_np = bboxes[:, :4].cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    scale = scale.cpu().numpy()
                    pad = pad.cpu().numpy()
                    bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                    bboxes_np[:, [1, 3]] -= pad[0]
                    bboxes_np[:, [0, 2]] /= scale[1]
                    bboxes_np[:, [1, 3]] /= scale[0]

                    # clip boxes
                    bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                    bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)
                    outputs.append({"boxes": torch.tensor(bboxes_np), "labels": labels, "scores": bboxes[:, 4]})
                return losses, outputs
            else:
                return losses
