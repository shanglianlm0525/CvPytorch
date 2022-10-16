# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/5 16:59
# @Author : liumin
# @File : detail_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class DetailAggregateLoss(nn.Module):
    def __init__(self, loss_weight=1.0, bce_loss_weight=1.0, dice_loss_weight=1.0, boundary_threshold=0.1, loss_name='detail_agg_loss'):
        super(DetailAggregateLoss, self).__init__()
        self.loss_weight = loss_weight
        self.bce_loss_weight = bce_loss_weight
        self.dice_loss_weight = dice_loss_weight
        self.boundary_threshold = boundary_threshold
        self._loss_name = loss_name
        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > self.boundary_threshold] = 1
        boundary_targets[boundary_targets <= self.boundary_threshold] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > self.boundary_threshold] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= self.boundary_threshold] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > self.boundary_threshold] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= self.boundary_threshold] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > self.boundary_threshold] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= self.boundary_threshold] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > self.boundary_threshold] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= self.boundary_threshold] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return self.loss_weight * (self.bce_loss_weight * bce_loss + self.dice_loss_weight * dice_loss)

    @property
    def loss_name(self):
        """Loss Name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name