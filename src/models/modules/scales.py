# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/29 9:22
# @Author : liumin
# @File : scales.py

import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    A learnable scale parameter
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale
