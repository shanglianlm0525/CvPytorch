# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/24 17:55
# @Author : liumin
# @File : cls_model.py

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

available_models = ['vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn'
                    ,'resnet18','resnet34','resnet50','resnet101','resnet152'
                    ,'resnext50_32x4d','resnext101_32x8d'
                    ,'densenet121','densenet161','densenet169','densenet201'
                    ,'shufflenet_v2_x0_5','shufflenet_v2_x1_0,','shufflenet_v2_x1_5','shufflenet_v2_x2_0'
                    ,'mobilenet_v2'
                    ,'squeezenet1_1']

class ClsModel(nn.Module):
    def __init__(self,dictionary):
        super(ClsModel, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1,3,224,224)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self._model = torchvision.models.__dict__['resnet18'](pretrained=True)
        num_fc_fea = self._model.fc.in_features
        self._model.fc = nn.Linear(num_fc_fea, self.num_classes)

        self.softmax =  nn.Softmax(dim=1)
        self._criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(self.weight)).float(),ignore_index=-1).cuda()

        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        outputs = self._model(imgs)

        if mode == 'infer':
            out = self.softmax(outputs)
            _, preds = torch.max(outputs, 1)

            return out
        else:
            losses = {}
            losses['loss'] = self._criterion(outputs, targets)

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
