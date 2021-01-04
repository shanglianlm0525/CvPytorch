# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/29 10:49
# @Author : liumin
# @File : atss.py

import torch
import torch.nn as nn

"""
    Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection
    https://arxiv.org/abs/1912.02424.pdf
"""

class ATSS(nn.Module):
    def __init__(self):
        super(ATSS, self).__init__()

    def forward(self, x):
        pass