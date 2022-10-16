# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/1 19:31
# @Author : liumin
# @File : dice_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
        https://arxiv.org/pdf/1606.04797.pdf
    """
    def __init__(self, weight=None, smooth=1., exponent=2, reduction='mean', ignore_index=255, loss_weight=1.0, loss_name='dice_loss'):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0] and pred.shape[2:] == target.shape[1:]
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        one_hot_target = F.one_hot(torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes)
        valid_mask = (target != self.ignore_index).long()

        return self.loss_weight * self.dice_loss(pred, one_hot_target, valid_mask)

    def dice_loss(self, pred, one_hot_target, valid_mask):
        assert pred.shape[0] == one_hot_target.shape[0]
        total_loss = 0
        num_classes = pred.shape[1]
        for i in range(num_classes):
            if i != self.ignore_index:
                dice_loss = binary_dice_loss(
                    pred[:, i], one_hot_target[..., i], valid_mask=valid_mask)
                if self.weight is not None:
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
        return total_loss / num_classes


    def binary_dice_loss(self, pred, target, valid_mask):
        assert pred.shape[0] == target.shape[0]
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + self.smooth
        den = torch.sum(pred.pow(self.exponent) + target.pow(self.exponent), dim=1) + self.smooth
        return 1 - num / den


    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name