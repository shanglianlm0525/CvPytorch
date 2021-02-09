# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/3 9:35
# @Author : liumin
# @File : seg_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .lovasz_losses import lovasz_softmax


def make_one_hot(labels, classes, ignore_index=None):
    if ignore_index is not None:
        ignore_i = labels == ignore_index
        # preset the element to be ignored to 0
        labels[ignore_i] = 0
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        # remove ignore index
        ignore_i = ignore_i.expand(ignore_i.shape[0], classes, ignore_i.shape[2], ignore_i.shape[3])
        target[ignore_i] = 0
    else:
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
    return target


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[2:]
        target = target.squeeze(1).long()
        return self.CE(output, target)


class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, pos_weight=None, ignore_index=255, reduction='mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[2:]
        target = make_one_hot(target.long(), output.size()[1], self.ignore_index)
        if self.weight is not None and self.weight.shape != target.shape:
            self.weight = self.weight.view(1, -1, 1, 1).expand(target.shape).cuda()
        if self.pos_weight is not None and self.pos_weight.shape != target.shape:
            self.pos_weight = self.pos_weight.view(1, -1, 1, 1).expand(target.shape).cuda()
        loss = F.binary_cross_entropy_with_logits(output, target.float(), weight=self.weight, pos_weight=self.pos_weight,
                                                 reduction=self.reduction)
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[2:]
        target = make_one_hot(target.long(), output.size()[1], self.ignore_index)
        output = self.softmax(output)
        target = target.float()
        # have to be contiguous since they may from a torch.view op
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.CE = CrossEntropyLoss2d(weight=alpha,ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        logpt = self.CE(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        return loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, weight=None, ignore_index=255, reduction='mean'):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.ce = CrossEntropyLoss2d(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        dice_loss = self.dice(output, target)
        return ce_loss + dice_loss

if __name__ == '__main__':
    input = torch.randn(4, 5, 6, 7)
    target = torch.randn(4, 5, 6, 7)

    ce_loss = CrossEntropyLoss2d()
    output = ce_loss(input, target)
    print('ce_loss', output.item())


    target = torch.ones([5, 6], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.full([5, 6], 1.5)  # A prediction (logit)
    pos_weight = torch.ones([6])  # All weights are equal to 1
    criterion = BCEWithLogitsLoss2d(pos_weight=pos_weight)
    criterion(output, target)  # -log(sigmoid(1.5))
    print('bce_loss', output)
