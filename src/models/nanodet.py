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
from .backbones.nanodet_shufflenet import ShuffleNetV2
from .heads.nanodet_head import NanoDetHead
from .necks import PAN



def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


class NanoDet(nn.Module):
    def __init__(self, dictionary=None):
        super(NanoDet, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 480, 360)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]
        '''
        backbone_cfg = {
            'model_size': '1.0x',
            'out_stages': [2,3,4],
            'activation': 'LeakyReLU'
        }
        self.backbone = ShuffleNetV2(**backbone_cfg)
        '''

        backbone_cfg = {'name': 'ShuffleNetV2', 'subtype': 'shufflenetv2_x1.0', 'out_stages': [2, 3, 4], 'output_stride': 32, 'pretrained': True}
        self.backbone = build_backbone(backbone_cfg)

        fpn_cfg = {
            'in_channels': [116, 232, 464],
            'out_channels': 96,
            # 'start_level': 0,
            # 'num_outs': 3
        }
        self.fpn = PAN(**fpn_cfg)
        head_cfg = {
              'num_classes': self.num_classes,
              'input_channel': 96,
              'feat_channels': 96,
              'stacked_convs': 2,
              'share_cls_reg': True,
              'octave_base_scale': 5,
              'scales_per_octave': 1,
              'strides': [8, 16, 32],
              'reg_max': 7,
              'norm_cfg':
                  {'type': 'BN'},
            'loss':{
                'loss_qfl':
                {'name': 'QualityFocalLoss',
                'use_sigmoid': True,
                'beta': 2.0,
                'loss_weight': 1.0
                 },
                'loss_dfl':
                {'name': 'DistributionFocalLoss',
                'loss_weight': 0.25},
                'loss_bbox':
                {'name': 'GIoULoss',
                'loss_weight': 2.0}
                }}
        self.head = NanoDetHead(**head_cfg)

        # self.init_params()

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
        new_warp_matrixs = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_boxes.append(target['boxes'])
            new_labels.append(target['labels'])
            new_warp_matrixs.append(target['warp_matrix'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["boxes"] = new_boxes
        t_targets["labels"] = new_labels
        t_targets["warp_matrix"] = new_warp_matrixs
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

            preds = self.head(self.fpn(self.backbone(imgs)))

            loss, loss_states = self.head.loss(preds, targets)

            losses['qfl_loss'] = loss_states["loss_qfl"]
            losses['bbox_loss'] = loss_states["loss_bbox"]
            losses['dfl_loss'] = loss_states["loss_dfl"]
            losses['loss'] = loss

            if mode == 'val':
                outputs = []
                dets = self.head.post_process(preds, imgs)
                # print('dets', dets)
                for warp_matrix, width, height, det in zip(targets['warp_matrix'], targets['width'], targets['height'], dets):
                    det_bboxes, det_labels = det
                    det_bboxes_np = det_bboxes[:, :4].cpu().numpy()
                    warp_matrix = warp_matrix.cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    det_bboxes_np = warp_boxes(det_bboxes_np, np.linalg.inv(warp_matrix), width, height)
                    outputs.append({"boxes": torch.tensor(det_bboxes_np), "labels": det_labels, "scores": det_bboxes[:, 4]})

                return losses, outputs
            else:
                return losses

    def test234(self, result_list, meta):
        preds = {}
        warp_matrix = meta['warp_matrix'][0] if isinstance(meta['warp_matrix'], list) else meta['warp_matrix']
        img_height = meta['img_info']['height'].cpu().numpy() \
            if isinstance(meta['img_info']['height'], torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() \
            if isinstance(meta['img_info']['width'], torch.Tensor) else meta['img_info']['width']
        for result in result_list:
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
            classes = det_labels.cpu().numpy()
            for i in range(self.num_classes):
                inds = (classes == i)
                preds[i] = np.concatenate([
                    det_bboxes[inds, :4].astype(np.float32),
                    det_bboxes[inds, 4:5].astype(np.float32)], axis=1).tolist()
        return preds