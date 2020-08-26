# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/17 14:25
# @Author : liumin
# @File : maskrcnn.py

import torch
from torch import nn
import torchvision

import numpy as np
from PIL import Image
from torchvision import models as modelsT
from torchvision import transforms as transformsT
import torch.nn.functional as torchF

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.utils.metrics import SegmentationEvaluator


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

class MaskRCNN(nn.Module):
    def __init__(self, dictionary=None):
        super(MaskRCNN, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 800)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        # load an instance segmentation model pre-trained pre-trained on COCO
        self._model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self._model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self._num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self._model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self._model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           self._num_classes)

        self.segmentationEvaluator = SegmentationEvaluator(self._num_classes)


    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, imgs, labels=None, mode='infer', **kwargs):

        threshold = 0.5
        if mode == 'infer':
            outputs = self._model(imgs)[0]
            boxes = outputs['boxes'].cpu().detach().numpy()
            scores = outputs['scores'].cpu().detach().numpy()
            lbls = outputs['labels'].cpu().detach().numpy()
            masks = outputs['masks'].cpu().detach().numpy()

            return
        else:
            device_id = imgs[0].data.device if isinstance(imgs, list) else imgs.data.device

            if mode == 'val':
                outputs = self._model(imgs)
                performances = {}
                performances['performance'] = 0

                print(outputs)
                print(labels)

                for output,label_gt in zip(outputs,labels):
                    boxes = output['boxes'].cpu().detach().numpy()
                    scores = output['scores'].cpu().detach().numpy()
                    lbls = output['labels'].cpu().detach().numpy()
                    masks = output['masks'].cpu().detach().numpy()

                    for box, score, lbl, mask in zip(boxes, scores, lbls, masks):
                        if score >= scores[0]:
                            print(box, score)




                    masks = outputs['masks'].cpu().detach().numpy()
                labels = np.array([label['masks'].unsqueeze(1).cpu().detach().numpy() for label in labels]) if isinstance(labels,list) else labels.cpu().numpy()
                print('masks.shape',masks.shape)
                print('labels[0].shape', labels[0].shape)
                preds = np.argmax(masks, axis=1)

                mask = (labels >= 0) & (labels < self._num_classes)
                label = self._num_classes * labels[mask].astype('int') + preds[mask]
                count = np.bincount(label, minlength=self.num_class ** 2)
                confusion_matrix = count.reshape(self._num_classes, self._num_classes)

                print('confusion_matrix',confusion_matrix)


                losses = {}
                losses['loss'] = torch.as_tensor(0,device=device_id)
                return losses, performances
            else:
                losses = self._model(imgs, labels)
                losses['loss'] = sum(loss for loss in losses.values())

                return losses


