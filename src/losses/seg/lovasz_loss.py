# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/1 19:32
# @Author : liumin
# @File : lovasz_loss.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import filterfalse


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_grad(gt_sorted):
    """Computes gradient of the Lovasz extension w.r.t sorted errors.

    See Alg. 1 in paper.
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def flatten_binary_logits(logits, labels, ignore_index=None):
    """Flattens predictions in the batch (binary case) Remove labels equal to
    'ignore_index'."""
    logits = logits.view(-1)
    labels = labels.view(-1)
    if ignore_index is None:
        return logits, labels
    valid = (labels != ignore_index)
    vlogits = logits[valid]
    vlabels = labels[valid]
    return vlogits, vlabels


def flatten_probs(probs, labels, ignore_index=None):
    """Flattens predictions in the batch."""
    if probs.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probs.size()
        probs = probs.view(B, 1, H, W)
    B, C, H, W = probs.size()
    probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
    labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = (labels != ignore_index)
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss.

    Args:
        logits (torch.Tensor): [P], logits at each prediction
            (between -infty and +infty).
        labels (torch.Tensor): [P], binary ground truth labels (0 or 1).

    Returns:
        torch.Tensor: The calculated loss.
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def lovasz_softmax_flat(probs, labels, classes='present', class_weight=None):
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): [P, C], class probabilities at each prediction
            (between 0 and 1).
        labels (torch.Tensor): [P], ground truth labels (between 0 and C - 1).
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        class_weight (list[float], optional): The weight for each class.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss.
    """
    if probs.numel() == 0:
        # only void pixels, the gradients should be 0
        return probs * 0.
    C = probs.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes == 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probs[:, 0]
        else:
            class_pred = probs[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        loss = torch.dot(errors_sorted, lovasz_grad(fg_sorted))
        if class_weight is not None:
            loss *= class_weight[c]
        losses.append(loss)
    return torch.stack(losses).mean()


class LovaszHingeLoss(nn.Module):
    """
        The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks
        https://arxiv.org/pdf/1705.08790.pdf

        Args:
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
    """

    def __init__(self, weight=None, classes='present', per_image=False, reduction='mean', ignore_index=255,
                 loss_weight=1.0, loss_name='lovasz_hinge_loss'):
        super(LovaszHingeLoss, self).__init__()
        self.weight = weight
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        return self.loss_weight * self.lovasz_hinge(pred, target)

    def lovasz_hinge(self, pred, target):
        if self.per_image:
            loss = mean([lovasz_hinge_flat(*flatten_binary_logits(
                    p.unsqueeze(0), t.unsqueeze(0), self.ignore_index))
                for p, t in zip(pred, target)
            ])
        else:
            loss = lovasz_hinge_flat(
                *flatten_binary_logits(pred, target, self.ignore_index))
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


class LovaszSoftmaxLoss(nn.Module):
    """
        The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks
        https://arxiv.org/pdf/1705.08790.pdf

        Args:
        classes (str | list[int], optional): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Default: 'present'.
        per_image (bool, optional): If per_image is True, compute the loss per
            image instead of per batch. Default: False.
    """
    def __init__(self, weight=None, classes='present', per_image=False, reduction='mean', ignore_index=255,
                 loss_weight=1.0, loss_name='lovasz_softmax_loss'):
        super(LovaszSoftmaxLoss, self).__init__()
        self.weight = weight
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self, pred, target):
        prob = F.softmax(pred, dim=1)
        return self.loss_weight * self.lovasz_softmax(prob, target)

    def lovasz_softmax(self, prob, target):
        if self.per_image:
            loss = mean([
                lovasz_softmax_flat(
                    *flatten_probs(p.unsqueeze(0), t.unsqueeze(0), self.ignore_index),
                    classes=self.classes, class_weight=self.weight)
                for p, t in zip(prob, target)
            ])
        else:
            loss = lovasz_softmax_flat(*flatten_probs(prob, target, self.ignore_index),
                classes=self.classes, class_weight=self.weight)
        return loss

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name