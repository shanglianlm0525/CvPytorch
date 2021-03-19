# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/11 12:49
# @Author : liumin
# @File : maskrcnn.py

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

"""
    Mask R-CNN
    https://arxiv.org/pdf/1703.06870.pdf
"""

class MaskRCNN(nn.Module):
    def __init__(self, dictionary=None):
        super(MaskRCNN, self).__init__()

        self.dictionary = dictionary
        self.input_size = [512, 512]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        # load an instance segmentation model pre-trained pre-trained on COCO
        self.model = maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.num_classes)


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
