# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/23 15:48
# @Author : liumin
# @File : yolo_fastestv2_head.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOFastestv2Head(nn.Module):
    def __init__(self, num_classes=80, input_channel=72, num_anchors=3):
        super(YOLOFastestv2Head, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_anchors = num_anchors  # number of anchors

        self.output_reg_layers = nn.Conv2d(input_channel, 4 * self.num_anchors, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(input_channel, self.num_anchors, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(input_channel, self.num_classes, 1, 1, 0, bias=True)

    def forward(self, x):
        reg_outs, cls_outs, obj_outs = x
        for i in range(len(x)):
            reg_outs[i] = self.output_reg_layers(reg_outs[i])
            cls_outs[i] = self.output_cls_layers(cls_outs[i])
            obj_outs[i] = self.output_obj_layers(obj_outs[i])
        return reg_outs, cls_outs, obj_outs