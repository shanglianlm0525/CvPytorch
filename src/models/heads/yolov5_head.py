# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/7 16:17
# @Author : liumin
# @File : yolov5_head.py

import torch
import torch.nn as nn


class YOLOv5Head(nn.Module):
    def __init__(self, channels=[256, 512, 1024], anchors=[], num_classes=80):
        super(YOLOv5Head, self).__init__()

        self.num_classes = num_classes  # number of classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.num_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.convs = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in channels)  # output conv

    def train(self, mode=True):
        super().train(mode=True)
        self.mode = 'train'
        print('self.mode = {self.mode}')

    def eval(self):
        super().train(mode=False)
        self.mode = 'val'
        print('self.mode = {self.mode}')

    def forword(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.num_layers):
            x[i] = self.convs[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.mode != 'train':  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_outputs))

        return x if self.mode=='train' else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()