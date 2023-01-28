# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/18 21:22
# @Author : liumin
# @File : airdet.py

import time

import torch
import torch.nn as nn
import numpy as np
import torchvision

from src.models.backbones import build_backbone
from src.models.heads.gflv2_head import postprocess_gfocal
from src.models.necks import build_neck
from src.models.heads import build_head
from src.models.detects import build_detect
from src.losses import build_loss

"""
    https://github.com/tinyvision/AIRDet
"""


class AIRDet(nn.Module):
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, dictionary=None, model_cfg=None):
        super(AIRDet, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.depth_mul, self.width_mul = self.cfg[self.model_cfg.TYPE.split("_")[1]]
        self.setup_extra_params()
        # backbone
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        # neck
        self.neck = build_neck(self.model_cfg.NECK)
        # head
        self.head = build_head(self.model_cfg.HEAD)

        self.conf_thres = 0.05  # confidence threshold
        self.nms_thres = 0.7  # NMS IoU threshold

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def setup_extra_params(self):
        self.model_cfg.BACKBONE.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.BACKBONE.__setitem__('width_mul', self.width_mul)
        self.model_cfg.NECK.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.NECK.__setitem__('width_mul', self.width_mul)
        self.model_cfg.HEAD.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.HEAD.__setitem__('width_mul', self.width_mul)
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)
        # self.model_cfg.LOSS.__setitem__('anchors', self.anchors)
        # self.model_cfg.LOSS.__setitem__('num_classes', self.num_classes)


    def trans_specific_format(self, imgs, targets):
        new_boxes = []
        new_labels = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for i, target in enumerate(targets):
            new_boxes.append(target['boxes'])
            new_labels.append(target['labels'])
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            if target.__contains__('height'):
                new_heights.append(target['height'])
            if target.__contains__('width'):
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
        if mode == 'infer':
            ''' for inference mode, img should preprocessed before feeding in net '''
            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)
            b, _, height, width = imgs.shape
            # imgs N x 3 x 640 x 640
            losses = {}
            features = self.neck(self.backbone(imgs))

            if mode == 'val':
                losses['loss'] = torch.tensor(0, device=imgs.device)

                outputs = []
                out = self.head(features, targets["boxes"], targets["labels"], imgs)
                if out is not None:
                    print('out', out[..., :10])
                    preds = postprocess_gfocal(out, self.conf_thres, self.nms_thres)
                    print('preds' ,preds)
                    for i, (width, height, scale, pad, pred) in enumerate(
                            zip(targets['width'], targets['height'], targets['scales'], targets['pads'], preds)):
                        scale = scale.cpu().numpy()
                        pad = pad.cpu().numpy()
                        width = width.cpu().numpy()
                        height = height.cpu().numpy()
                        predn = pred.clone()

                        bboxes_np = predn[:, :4].cpu().numpy()
                        bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                        bboxes_np[:, [1, 3]] -= pad[0]
                        bboxes_np[:, [0, 2]] /= scale[1]
                        bboxes_np[:, [1, 3]] /= scale[0]

                        # clip boxes
                        bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                        bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)
                        outputs.append({"boxes": torch.tensor(bboxes_np), "labels": pred[:, 5], "scores": pred[:, 4]})
                        # outputs.append({"boxes": pred[:, :4], "labels": pred[:, 5], "scores": pred[:, 4]})
                return losses, outputs
            else:
                loss_states = self.head(features, targets["boxes"], targets["labels"], imgs)
                # print(loss_states)
                losses['loss'] = loss_states['total_loss']
                losses['loss_cls'] = loss_states['loss_cls']
                losses['bbox_loss'] = loss_states['loss_bbox']
                losses['dfl_loss'] = loss_states['loss_dfl']
                return losses