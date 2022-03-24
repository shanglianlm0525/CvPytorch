# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/24 12:59
# @Author : liumin
# @File : giraffedet.py

import torch
import torch.nn as nn


"""
    GiraffeDet: A Heavy-Neck Paradigm for Object Detection
    https://arxiv.org/pdf/2202.04256.pdf
"""





class GiraffeDet(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(GiraffeDet, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 800, 800)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]



    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05