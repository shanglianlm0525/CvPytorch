# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/18 16:31
# @Author : liumin
# @File : yolov4_neck.py

import torch
import torch.nn as nn


class YOLOV4Neck(nn.Module):
    ''''''

    def __init__(self, features=256):
        super().__init__()

    def forward(self, x):
        pass