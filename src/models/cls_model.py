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

    def forward(self, img, labels=None, mode='infer', **kwargs):
        outputs = self._model(img)

        if mode == 'infer':
            out = self.softmax(outputs)
            _, preds = torch.max(outputs, 1)

            return out
        else:
            losses = {}
            loss = self._criterion(outputs, labels)
            device_id = labels.data.device

            losses['all_loss'] = loss
            losses['all_loss_num'] = torch.as_tensor(labels.size(0),device=device_id)

            if mode == 'val':
                performances = {}
                _, preds = torch.max(outputs, 1)
                correct_num = preds.eq(labels).sum()
                performances['all_perf'] = torch.as_tensor(correct_num,device=device_id)
                performances['all_perf_num'] = torch.as_tensor(labels.size(0),device=device_id)

                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = labels == idx
                        if labels[cognize].size(0):
                            losses[_label + '_loss'] = torchF.cross_entropy(outputs[cognize], labels[cognize]) * _weight
                            losses[_label + '_loss_num'] = torch.as_tensor(labels[cognize].size(0),device=device_id)
                            performances[_label + '_perf'] = torch.as_tensor(preds[cognize].eq(labels[cognize]).sum(),device=device_id)
                            performances[_label + '_perf_num'] = torch.as_tensor(labels[cognize].size(0),device=device_id)
                        else:
                            losses[_label + '_loss'] = torch.as_tensor(0,device=device_id)
                            losses[_label + '_loss_num'] = torch.as_tensor(0,device=device_id)
                            performances[_label + '_perf'] = torch.as_tensor(0,device=device_id)
                            performances[_label + '_perf_num'] = torch.as_tensor(0,device=device_id)

                return losses, performances
            else:
                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = labels == idx
                        if labels[cognize].size(0):
                            losses[_label + '_loss'] = torchF.cross_entropy(outputs[cognize], labels[cognize]) * _weight
                            losses[_label + '_loss_num'] = torch.as_tensor(labels[cognize].size(0),device=device_id)
                        else:
                            losses[_label + '_loss'] = torch.as_tensor(0,device=device_id)
                            losses[_label + '_loss_num'] = torch.as_tensor(0,device=device_id)
                return losses



