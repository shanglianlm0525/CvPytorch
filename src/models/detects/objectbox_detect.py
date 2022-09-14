# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/26 20:27
# @Author : liumin
# @File : objectbox_detect.py

import math
import torch
import torch.nn as nn
import numpy as np


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class ObjectBoxDetect(nn.Module):
    det_layers_t = [[[1.0000, 1.0000]],
            [[0.5000, 0.5000]],
            [[0.2500, 0.2500]]]
    def __init__(self, num_classes=80, in_channels=[256, 512, 1024], stride=[8, 16, 32], depth_mul=1.0, width_mul=1.0):  # detection layer
        super().__init__()
        self.num_classes = num_classes  # number of classes
        self.num_outputs = num_classes + 5  # number of outputs per detection layers
        self.num_layers = len(self.det_layers_t)  # number of detection layers
        self.stride = stride
        self.num_anchors = 1
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))

        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        # a = torch.tensor(self.det_layers1).float().view(self.num_layers, -1, 2)[:, :self.num_anchors, :]
        # a /= torch.tensor(self.stride).view(-1, 1, 1)
        a = torch.tensor(self.det_layers_t)
        self.register_buffer('det_layers', a)  # shape(nl,na,2)
        self.register_buffer('det_layers_grid', a.clone().view(self.num_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in self.in_channels)  # output conv

        self.init_weights()

    def init_weights(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                coord = torch.ones([bs, self.num_anchors, ny, nx], device=y.device)
                coord = coord.nonzero(as_tuple=False)
                x_coord = coord[:, 3]
                y_coord = coord[:, 2]

                power = 2 ** i

                s_gain = torch.ones_like(self.det_layers_grid[i, ..., 0]) * power
                dx1 = (y[..., 0] * 2) ** 2 * s_gain
                dy1 = (y[..., 1] * 2) ** 2 * s_gain
                dx2 = (y[..., 2] * 2) ** 2 * s_gain
                dy2 = (y[..., 3] * 2) ** 2 * s_gain

                y[..., 0] = (x_coord.view(bs, self.num_anchors, ny, nx) + 1 - (dx1)) * self.stride[i]
                y[..., 1] = (y_coord.view(bs, self.num_anchors, ny, nx) + 1 - (dy1)) * self.stride[i]
                y[..., 2] = (x_coord.view(bs, self.num_anchors, ny, nx) + (dx2)) * self.stride[i]
                y[..., 3] = (y_coord.view(bs, self.num_anchors, ny, nx) + (dy2)) * self.stride[i]

                xyxy = y[..., :4].view(-1, 4)
                xywh = xyxy2xywh(xyxy)
                y[..., :4] = xywh.view(bs, self.num_anchors, ny, nx, 4)

                pred = y.view(bs, -1, self.num_outputs)
                z.append(pred)

        return (None, x) if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
