# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/7 16:17
# @Author : liumin
# @File : yolov5_head.py

import math
import torch
import torch.nn as nn


class YOLOv5Head(nn.Module):
    def __init__(self, num_classes=80, channels=[256, 512, 1024], depth_mul=1.0, width_mul=1.0, stride=[ 8., 16., 32.]):
        super(YOLOv5Head, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.depth_mul = depth_mul
        self.width_mul = width_mul
        self.stride = stride
        self.channels = list(map(lambda x: max(round(x * self.width_mul), 1), channels))
