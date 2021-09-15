# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/7 16:45
# @Author : liumin
# @File : yolop_head.py

import math

import torch
import torch.nn as nn


from src.models.modules.yolov5_modules import Conv, BottleneckCSP


class SegmentHead(nn.Module):
    def __init__(self, num_classes=2):
        super(SegmentHead, self).__init__()
        self.segHead = nn.Sequential(
            Conv(256, 128, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            BottleneckCSP(128, 64, n=1, shortcut=False),
            Conv(64, 32, 3, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(32, 16, 3, 1),
            BottleneckCSP(16, 8, n=1, shortcut=False),
            nn.Upsample(scale_factor=2, mode='nearest'),
            Conv(8, num_classes, 3, 1)
        )

    def forward(self, x):
        return self.segHead(x)


class YOLOPHead(nn.Module):
    def __init__(self, num_classes=80, channels=[128, 256, 512], stride=[ 8., 16., 32.], anchors=[[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]):
        super(YOLOPHead, self).__init__()
        self.num_classes = num_classes

        # self.detect_head = YOLOv5Head(num_classes=num_classes, channels=channels, depth_mul=1.0, width_mul=1.0, stride=stride, anchors=anchors)
        self.drivable_area_segment_head = SegmentHead()
        self.lane_line_segment_head = SegmentHead()

    def forward(self, x):
        seg_x, det_x = x
        det_out, det_train_out = self.detect_head(det_x)
        seg_drivable_area = self.drivable_area_segment_head(seg_x)
        seg_lane_line = self.lane_line_segment_head(seg_x)
        return det_out, det_train_out, seg_drivable_area, seg_lane_line


