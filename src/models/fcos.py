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

        # mode == "infer":
        self.detection_head = DetectHead(0.05, 0.6, 1000, [8,16,32,64,128])
        self.clip_boxes = ClipBoxes()

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
            batch_boxes, batch_classes = torch.split(targets, 4, 2)
            batch_classes = batch_classes.squeeze(2)
            out = self.fcos_body(imgs)
            predicts = self.target_layer([out, batch_boxes, batch_classes])
            loss_tuple = self.loss_layer([out, predicts])

            losses['cls_loss'] = loss_tuple[0]
            losses['cnt_loss'] = loss_tuple[1]
            losses['reg_loss'] = loss_tuple[2]
            losses['loss'] = loss_tuple[-1]

            if mode == 'val':
                scores, classes, boxes = self.detection_head(out)
                boxes = self.clip_boxes(imgs, boxes)
                '''
                print('targets',targets.shape)
                print('scores',scores.shape)
                print('classes',classes.shape)
                print('boxes',boxes.shape)
                '''
                # performances['performance'] = 1000-losses['loss']
                return losses, (classes, scores, boxes)
            else:
                return losses
