# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/7 16:17
# @Author : liumin
# @File : yolov5_head.py

import torch
import torch.nn as nn


class YOLOv5Head(nn.Module):
    def __init__(self):
        super(YOLOv5Head, self).__init__()

    def forword(self, x):
        pass