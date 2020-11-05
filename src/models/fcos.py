# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/2 18:47
# @Author : liumin
# @File : fcos.py

import torch
import torch.nn as nn
from .tools.fcos_detect import FcosBody, GenTargets, ClipBoxes, DetectHead
from ..losses.fcos_loss import ClsCntRegLoss


class FCOS(nn.Module):
    def __init__(self, dictionary=None):
        super(FCOS, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.fcos_body = FcosBody(self._num_classes)

        # mode == "train or val":
        self.target_layer = GenTargets()
        self.loss_layer = ClsCntRegLoss()

        # inference
        score_threshold = 0.05
        nms_iou_threshold = 0.6
        max_detection_boxes_num = 1000
        # mode == "infer":
        self.detection_head = DetectHead(0.05, 0.6,1000, [8,16,32,64,128])
        self.clip_boxes = ClipBoxes()

    def forward(self, inputs, mode='infer'):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode == "train":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == "infer":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes