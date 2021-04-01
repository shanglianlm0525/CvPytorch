# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/1 9:29
# @Author : liumin
# @File : plug_and_play.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
        Spatial Transformer Networks
        https://arxiv.org/pdf/1506.02025.pdf
    """
    def __init__(self, spatial_dims):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims
        self.fc1 = nn.Linear(32*4*4, 1024) # 可根据自己的网络参数具体设置
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        batch_images = x #保存一份原始数据
        x = x.view(-1, 32*4*4)
        # 利用FC结构学习到6个参数
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 2,3) # 2x3
        # 利用affine_grid生成采样点
        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))
        # 将采样点作用到原始数据上
        rois = F.grid_sample(batch_images, affine_grid_points)
        return rois, affine_grid_points



