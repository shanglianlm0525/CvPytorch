# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/1 18:33
# @Author : liumin
# @File : cross_entropy_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, reduction='mean', pos_weight=None, loss_weight=1.0, loss_name='bce(w/sigmoid)_loss'):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.criterion = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction, pos_weight=pos_weight)

    def forward(self, pred, target):
        return self.loss_weight * self.criterion(pred, target.long())

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name



class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean', label_smoothing=0.0, loss_weight=1.0, loss_name='ce_loss'):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, pred, target):
        return self.loss_weight * self.criterion(pred, target.long())

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class OhemCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, thresh=0.7, min_kept=100000, ignore_index=255, reduction='none',
                 label_smoothing=0.0, loss_weight=1.0, loss_name='ohem_ce_loss'):
        super(OhemCrossEntropyLoss2d, self).__init__()
        self.loss_weight = loss_weight
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.min_kept = min_kept
        self._loss_name = loss_name
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction,
                                             label_smoothing=label_smoothing)

    def forward(self, pred, target):
        loss = self.loss_weight * self.criterion(pred, target.long()).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name