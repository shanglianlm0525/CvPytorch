# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/9 13:21
# @Author : liumin
# @File : yolop_loss.py

import torch
import torch.nn as nn

from src.losses.yolov5_loss import Yolov5Loss


class YolopLoss(nn.Module):
    def __init__(self, num_classes=80, num_layers = 3, num_anchors = 3, stride = [8., 16., 32.], anchors = [[3, 9, 5, 11, 4, 20], [7, 18, 6, 39, 12, 31], [19, 50, 38, 81, 68, 157]], device = 'cuda:0'):
        super(YolopLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        self.detection_criterion = Yolov5Loss(num_classes, num_layers, num_anchors, stride, anchors)
        self.drivable_area_segmentation_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.lane_line_segmentation_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.lane_line_segmentation_iou_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))

    def forward(self, preds, gts):
        det_loss, box_loss, obj_loss, cls_loss = self.detection_criterion(preds, gts)
        seg_da_ce = self.drivable_area_segmentation_criterion()
        seg_ll_ce = self.lane_line_segmentation_criterion()
        seg_iou_ll = self.lane_line_segmentation_iou_criterion()
        loss = box_loss + obj_loss + cls_loss + seg_da_ce + seg_ll_ce + seg_iou_ll
        return loss, box_loss, obj_loss, cls_loss, seg_da_ce, seg_ll_ce, seg_iou_ll