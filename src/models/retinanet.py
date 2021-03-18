# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/18 12:48
# @Author : liumin
# @File : retinanet.py


import torch
import torch.nn as nn
import torchvision
from torchvision import ops
from torchvision.models import detection
from torchvision.models.detection import retinanet_resnet50_fpn
"""
    Focal Loss for Dense Object Detection
    https://arxiv.org/pdf/1708.02002.pdf
"""

class RetinaNet(nn.Module):
    def __init__(self, dictionary=None):
        super(RetinaNet, self).__init__()

        self.dictionary = dictionary
        self.input_size = [512, 512]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.model = retinanet_resnet50_fpn(pretrained=False, progress=True,
                                            num_classes=self.num_classes, pretrained_backbone=True, trainable_backbone_layers=None)


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.5
        if mode == 'infer':
            predicts = self.model(imgs)
            return predicts
        else:
            if mode == 'val':
                losses = {}
                losses['loss'] = 0
                outputs = self.model(imgs)
                return losses, outputs
            else:
                losses = self.model(imgs, targets)
                losses['loss'] = sum(loss for loss in losses.values())
                return losses