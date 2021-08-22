# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 13:48
# @Author : liumin
# @File : yolox_head.py
import math

import torch
import torch.nn as nn

from src.models.modules.yolox_modules import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes=80, strides=[8, 16, 32], channels=[256, 512, 1024], subtype='yolox_s', depthwise=False):
        super(YOLOXHead, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.strides = strides
        self.subtype = subtype
        self.n_anchors = 1
        self.grids = [torch.zeros(1)] * len(channels)

        if self.subtype == 'yolox_s':
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = [32, 64, 128, 256, 512]
        elif self.subtype == 'yolox_m':
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = [48, 96, 192, 384, 768]
        elif self.subtype == 'yolox_l':
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = [64, 128, 256, 512, 1024]
        elif self.subtype == 'yolox_x':
            dep_mul, wid_mul = 1.33, 1.25
            self.out_channels = [80, 160, 320, 640, 1280]
        else:
            raise NotImplementedError

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(channels)):
            self.stems.append(
                BaseConv(in_channels=int(channels[i] * wid_mul), out_channels=int(256 * wid_mul), ksize=1,stride=1)
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[ Conv(in_channels=int(256 * wid_mul),out_channels=int(256 * wid_mul),ksize=3,stride=1),
                        Conv(in_channels=int(256 * wid_mul),out_channels=int(256 * wid_mul),ksize=3,stride=1),]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[Conv(in_channels=int(256 * wid_mul),out_channels=int(256 * wid_mul),ksize=3,stride=1),
                        Conv(in_channels=int(256 * wid_mul),out_channels=int(256 * wid_mul),ksize=3,stride=1),]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(in_channels=int(256 * wid_mul),out_channels=self.n_anchors * self.num_classes,kernel_size=1,stride=1,padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channels=int(256 * wid_mul),out_channels=4,kernel_size=1,stride=1,padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels=int(256 * wid_mul),out_channels=self.n_anchors * 1,kernel_size=1,stride=1,padding=0)
            )

        self._init_weight()


    def _init_weight(self, prior_prob=1e-2):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        outputs = []
        preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(zip(self.cls_convs, self.reg_convs, self.strides, x)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            preds.append(torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1))

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, x[0].type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(
                torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(x[0])
            )
            outputs.append(output)

        # return preds, [outputs, x_shifts, y_shifts, expanded_strides]
        return preds, outputs, x_shifts, y_shifts, expanded_strides


    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = (
            output.permute(0, 1, 3, 4, 2)
                .reshape(batch_size, self.n_anchors * hsize * wsize, -1)
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

