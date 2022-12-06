# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/11 10:33
# @Author : liumin
# @File : yolov7_detect.py

import math
import torch
import torch.nn as nn

from src.models.modules.yolov7_modules import ImplicitA, ImplicitM


class YOLOv7Detect1(nn.Module):
    def __init__(self, num_classes=80, in_channels=[256, 512, 1024], stride=[8., 16., 32.], anchors=(), depth_mul=1.0, width_mul=1.0):  # detection layer
        super(YOLOv7Detect1, self).__init__()
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))
        self.num_classes = num_classes  # number of classes
        self.num_outputs = self.num_classes + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0])  # number of anchors
        self.stride = torch.tensor(stride)
        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', (a.clone() * self.stride.view(-1, 1, 1)).view(self.num_layers, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in in_channels)  # output conv

        self.ia = nn.ModuleList(ImplicitA(x) for x in in_channels)
        self.im = nn.ModuleList(ImplicitM(self.num_outputs * self.num_anchors) for _ in in_channels)

        self.init_weight()

    def init_weight(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=self.num_classes) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if cf is None else torch.log(
                cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.num_layers):
            x[i] = self.m[i](self.ia[i](x[i]))  # conv
            x[i] = self.im[i](x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_outputs))

        # return x if self.training else (torch.cat(z, 1), x)
        return (None, x) if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class YOLOv7Detect(nn.Module):
    def __init__(self, num_classes=80, in_channels=[256, 512, 1024], stride=[8., 16., 32.], anchors=(), depth_mul=1.0, width_mul=1.0):  # detection layer
        super(YOLOv7Detect, self).__init__()
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))

        self.num_classes = num_classes  # number of classes
        self.num_outputs = self.num_classes + 5  # number of outputs per anchor
        self.num_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0])  # number of anchors
        self.stride = torch.tensor(stride)

        self.grid = [torch.zeros(1)] * self.num_layers  # init grid
        a = torch.tensor(anchors).float().view(self.num_layers, -1, 2)
        self.register_buffer('anchors', a)  # shape(num_layers,num_anchors,2)
        self.register_buffer('anchor_grid', (a.clone() * self.stride.view(-1, 1, 1)).view(self.num_layers, 1, -1, 1, 1, 2))  # shape(num_layers,1,num_anchors,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.num_outputs * self.num_anchors, 1) for x in in_channels)  # output conv

        self.init_weight()

    def init_weight(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=self.num_classes) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.num_anchors, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.num_layers):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.num_outputs))

        # return x if self.training else (torch.cat(z, 1), x)
        return (None, x) if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()