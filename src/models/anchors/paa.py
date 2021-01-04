# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/29 10:47
# @Author : liumin
# @File : paa.py

import torch
import torch.nn as nn


"""
    Probabilistic Anchor Assignment with IoU Prediction for Object Detection
    https://arxiv.org/pdf/2007.08103.pdf
"""

class PAA(nn.Module):
    def __init__(self):
        super(PAA, self).__init__()

    def forward(self, x):
        pass