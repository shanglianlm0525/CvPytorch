# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/2 18:47
# @Author : liumin
# @File : fcos_old.py

import torch
import torch.nn as nn
import numpy as np
from .tools.fcos_detect import FcosBody, GenTargets, ClipBoxes, DetectHead
from ..losses.fcos_loss import ClsCntRegLoss

def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # nanodet_transform
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

class FCOS(nn.Module):
    def __init__(self, dictionary=None):
        super(FCOS, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self.num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.fcos_body = FcosBody(self.num_classes)

        loss_cfg = {'num_classes': 80, 'strides': [8, 16, 32], 'reg_max': 7, 'use_sigmoid': True}

        # mode == "train or val":
        self.target_layer = GenTargets()
        self.loss_layer = ClsCntRegLoss()

        # mode == "infer":
        self.detection_head = DetectHead(0.05, 0.6, 1000, [8,16,32,64,128])
        self.clip_boxes = ClipBoxes()

    def post_process(self, predicts, meta):
        scores, labels, boxes = predicts
        warp_matrix = meta['warp_matrix'][0] if isinstance(meta['warp_matrix'], list) else meta['warp_matrix']
        warp_matrix = warp_matrix.cpu().numpy() if isinstance(warp_matrix, torch.Tensor) else warp_matrix
        img_height = meta['img_info']['height'].cpu().numpy() if isinstance(meta['img_info']['height'],
                                                                            torch.Tensor) else meta['img_info']['height']
        img_width = meta['img_info']['width'].cpu().numpy() if isinstance(meta['img_info']['width'],
                                                                          torch.Tensor) else meta['img_info']['width']
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        boxes = boxes.detach().cpu().numpy()

        preds = {}
        for score, label, box in zip(scores, labels, boxes):
            box[:, :] = warp_boxes(box, np.linalg.inv(warp_matrix), img_width, img_height)

            for i in range(self.num_classes):
                inds = (label == i)
                preds[i] = np.concatenate([box[inds, :].astype(np.float32), np.expand_dims(score[inds],1).astype(np.float32)], axis=1).tolist()
        return preds

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
            out = self.fcos_body(imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(imgs, boxes)
            return scores, classes, boxes
        else:
            losses = {}
            batch_boxes, batch_classes = targets['gt_bboxes'], targets['gt_labels']

            out = self.fcos_body(imgs)
            predicts = self.target_layer([out, batch_boxes, batch_classes])
            loss_tuple = self.loss_layer([out, predicts])

            losses['cls_loss'] = loss_tuple[0]
            losses['cnt_loss'] = loss_tuple[1]
            losses['reg_loss'] = loss_tuple[2]
            losses['loss'] = loss_tuple[-1]

            if mode == 'val':
                scores, labels, boxes = self.detection_head(out)
                boxes = self.clip_boxes(imgs, boxes)
                preds = self.post_process([scores, labels, boxes], targets)
                return losses, preds
            else:
                return losses
