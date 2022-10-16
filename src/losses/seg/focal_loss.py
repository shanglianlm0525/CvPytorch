# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/1 19:31
# @Author : liumin
# @File : focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F



class FocalLoss(nn.Module):
    """
        Focal Loss for Dense Object Detection
        https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, weight=None, gamma=2.0, alpha=0.5, reduction='mean', ignore_index=255, label_smoothing=0.0, loss_weight=1.0, loss_name='focal_loss'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction,
                                             label_smoothing=label_smoothing)

    def forward(self, pred, target):
        if pred.dim() > 2:
            pred = pred.contiguous().view(pred.size(0), pred.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, pred.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.criterion(pred, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * self.alpha * logpt
        return self.loss_weight * loss

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name