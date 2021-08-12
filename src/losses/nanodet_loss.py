# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/29 9:32
# @Author : liumin
# @File : nanodet_loss.py

import functools

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from src.losses.det.general_focal_losses import QualityFocalLoss, DistributionFocalLoss
from src.losses.det.iou_losses import GIoULoss, bbox_overlaps
from src.models.assigners.atss import ATSS
from src.models.modules.nms import multiclass_nms




class NanoDetLoss(nn.Module):
    def __init__(self, loss):
        super(NanoDetLoss, self).__init__()
        self.loss_cfg = loss
        self.assigner = ATSS(topk=9)

        self.num_classes = self.loss_cfg['num_classes']
        self.strides = self.loss_cfg["strides"]
        self.grid_cell_scale = self.loss_cfg["octave_base_scale"]

        self.loss_qfl = QualityFocalLoss(beta=self.loss_cfg["loss_qfl"]["beta"],
                                         loss_weight=self.loss_cfg["loss_qfl"]["loss_weight"])
        self.loss_dfl = DistributionFocalLoss(loss_weight=self.loss_cfg["loss_dfl"]["loss_weight"])
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg["loss_bbox"]["loss_weight"])

    def forward(self, preds, gt_meta):
        pass
