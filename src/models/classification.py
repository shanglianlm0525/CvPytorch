# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/24 15:57
# @Author : liumin
# @File : classification.py

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from src.losses.seg_loss import CrossEntropyLoss2d
from src.models.backbones import build_backbone

available_models = ['vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn'
                    ,'resnet18','resnet34','resnet50','resnet101','resnet152'
                    ,'resnext50_32x4d','resnext101_32x8d'
                    ,'densenet121','densenet161','densenet169','densenet201'
                    ,'shufflenet_v2_x0_5','shufflenet_v2_x1_0,','shufflenet_v2_x1_5','shufflenet_v2_x2_0'
                    ,'mobilenet_v2'
                    ,'squeezenet1_1']



class Classification(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(Classification, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.input_size = [224, 224]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)

        self.criterion = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(self.weight)).float()).cuda()

    def setup_extra_params(self):
        self.model_cfg.BACKBONE.__setitem__('num_classes', self.num_classes)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        outputs = self.backbone(imgs)

        if mode == 'infer':
            out = self.softmax(outputs)
            _, preds = torch.max(outputs, 1)
            return out
        else:
            losses = {}
            losses['loss'] = self.criterion(outputs, targets)

            if mode == 'val':
                _, preds = torch.max(outputs, 1)
                return losses, preds
            else:
                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = targets == idx
                        if targets[cognize].size(0):
                            losses['loss_'+_label] = F.cross_entropy(outputs[cognize], targets[cognize]) * _weight

                return losses
