# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 16:15
# @Author : liumin
# @File : cls_model.py

import torch
import torch.nn as nn
from torchvision import models as modelsT
import numpy as np
import torch.nn.functional as torchF

class ClsModel(nn.Module):
    def __init__(self,dictionary):
        super(ClsModel, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1,3,224,224)

        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]
        self._num_classes = len(self.dictionary)

        self._model = modelsT.resnet18(pretrained=True)
        num_fc_fea = self._model.fc.in_features
        self._model.fc = nn.Linear(num_fc_fea, self._num_classes)

        self.softmax =  nn.Softmax(dim=1)

        self._criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(self._weight)).float(),ignore_index=-1).cuda()


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

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        outputs = self._model(imgs)

        if mode == 'infer':
            out = self.softmax(outputs)
            _, preds = torch.max(outputs, 1)

            return out
        else:
            losses = {}
            losses['loss'] = self._criterion(outputs, labels)

            if mode == 'val':
                performances = {}
                _, preds = torch.max(outputs, 1)
                performances['performance'] = preds.eq(labels).sum() *1.0 / imgs.size(0)

                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = labels == idx
                        if labels[cognize].size(0):
                            losses['loss_'+_label] = torchF.cross_entropy(outputs[cognize], labels[cognize]) * _weight
                            performances['performance_'+_label] = preds[cognize].eq(labels[cognize]).sum() * 1.0 / torch.nonzero(cognize,as_tuple=False).size(0)

                return losses, performances
            else:
                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = labels == idx
                        if labels[cognize].size(0):
                            losses['loss_'+_label] = torchF.cross_entropy(outputs[cognize], labels[cognize]) * _weight

                return losses



