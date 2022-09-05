# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/7 16:52
# @Author : liumin
# @File : det_target_transforms.py

import torch
import numpy as np

from src.models.anchors.prior_box import PriorBox


__all__ = ['Compose']


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample



