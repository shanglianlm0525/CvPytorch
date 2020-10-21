# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:03
# @Author : liumin
# @File : pspnet.py

"""
    SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
    https://arxiv.org/pdf/1511.00561.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from src.evaluation.eval_segmentation import SegmentationEvaluator