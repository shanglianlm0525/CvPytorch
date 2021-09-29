# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/28 11:11
# @Author : liumin
# @File : openpose_loss.py

from collections import OrderedDict
import torch
import torch.nn as nn


def build_names(layer_name):
    names = []
    for j in range(1, 7):
        for k in layer_name:
            names.append('loss_stage%d_%s' % (j, k))
    return names


class OpenPoseLoss:
    # Compute losses
    def __init__(self, layer_name=['heatmap', 'paf']):
        super(OpenPoseLoss, self).__init__()
        self.layer_name = layer_name
        self.loss_names = build_names(layer_name)
        self.criterion = nn.MSELoss(reduction='mean').cuda()

    def __call__(self, train_out, heat_gt, vec_gt):
        total_loss = 0
        loss_items = OrderedDict()
        for j, (heat, vec) in enumerate(zip(train_out[0], train_out[1])):
            # Compute losses
            loss_items['loss_stage%d_%s' % (j, self.layer_name[0])] = self.criterion(heat, heat_gt)
            loss_items['loss_stage%d_%s' % (j, self.layer_name[1])] = self.criterion(vec, vec_gt)

            total_loss += loss_items['loss_stage%d_%s' % (j, 'heatmap')]
            total_loss += loss_items['loss_stage%d_%s' % (j, 'paf')]

        return total_loss, loss_items