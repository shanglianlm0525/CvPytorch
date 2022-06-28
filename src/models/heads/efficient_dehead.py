# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/24 16:04
# @Author : liumin
# @File : efficient_dehead.py
import math

import torch
import torch.nn as nn

from src.models.modules.yolov6_modules import Conv


class EfficientDeHead(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolov6_s', num_classes=80, in_channels=[128, 256, 512], num_anchors=1, num_layers=3):  # detection layer
        super().__init__()
        self.subtype = subtype
        self.num_classes = num_classes  # number of classes
        self.num_outputs = num_classes + 5  # number of outputs per anchor
        self.num_layers = num_layers  # number of detection layers
        if isinstance(num_anchors, (list, tuple)):
            self.num_anchors = len(num_anchors[0]) // 2
        else:
            self.num_anchors = num_anchors
        self.num_anchors = num_anchors
        self.grid = [torch.zeros(1).cuda()] * num_layers
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride).cuda()

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        # Efficient decoupled head layers
        self.stems = nn.Sequential(
            # stem0
            Conv(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=1, stride=1),
            # stem1
            Conv(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=1, stride=1),
            # stem2
            Conv(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=1, stride=1)
        )
        self.cls_convs = nn.Sequential(
            # cls_conv0
            Conv(in_channels=in_channels[0], out_channels=in_channels[0], kernel_size=3, stride=1),
            # cls_conv1
            Conv(in_channels=in_channels[1],out_channels=in_channels[1],kernel_size=3,stride=1),
            # cls_conv2
            Conv(in_channels=in_channels[2],out_channels=in_channels[2],kernel_size=3,stride=1)
        )
        self.reg_convs = nn.Sequential(
            # reg_conv0
            Conv(in_channels=in_channels[0], out_channels=in_channels[0],kernel_size=3,stride=1),
            # reg_conv1
            Conv(in_channels=in_channels[1], out_channels=in_channels[1],kernel_size=3,stride=1),
            # reg_conv2
            Conv(in_channels=in_channels[2],out_channels=in_channels[2],kernel_size=3,stride=1)
        )
        self.cls_preds = nn.Sequential(
            # cls_pred0
            nn.Conv2d(in_channels=in_channels[0],out_channels=num_classes * num_anchors,kernel_size=1),
            # cls_pred1
            nn.Conv2d(in_channels=in_channels[1], out_channels=num_classes * num_anchors,kernel_size=1),
            # cls_pred2
            nn.Conv2d(in_channels=in_channels[2],out_channels=num_classes * num_anchors,kernel_size=1)
        )
        self.reg_preds = nn.Sequential(
            # reg_pred0
            nn.Conv2d(in_channels=in_channels[0],out_channels=4 * num_anchors,kernel_size=1),
            # reg_pred1
            nn.Conv2d(in_channels=in_channels[1],out_channels=4 * num_anchors,kernel_size=1),
            # reg_pred2
            nn.Conv2d(in_channels=in_channels[2],out_channels=4 * num_anchors,kernel_size=1)
        )
        self.obj_preds = nn.Sequential(
            # obj_pred0
            nn.Conv2d(in_channels=in_channels[0],out_channels=1 * num_anchors,kernel_size=1),
            # obj_pred1
            nn.Conv2d(in_channels=in_channels[1],out_channels=1 * num_anchors,kernel_size=1),
            # obj_pred2
            nn.Conv2d(in_channels=in_channels[2],out_channels=1 * num_anchors,kernel_size=1)
        )

        self.init_weights()

    def init_weights(self):
        prior_prob = 1e-2
        for conv in self.cls_preds:
            b = conv.bias.view(self.num_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        for conv in self.obj_preds:
            b = conv.bias.view(self.num_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        z = []
        for i in range(self.num_layers):
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            # training:
            x[i] = torch.cat([reg_output, obj_output, cls_output], 1)
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # inference
            y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            bs, _, ny, nx = y.shape
            y = y.view(bs, self.num_anchors, self.num_outputs, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if self.grid[i].shape[2:4] != y.shape[2:4]:
                d = self.stride.device
                yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing = 'ij')
                self.grid[i] = torch.stack((xv, yv), 2).view(1, self.num_anchors, ny, nx, 2).float()

            y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i] # wh
            z.append(y.view(bs, -1, self.num_outputs))
        # return x if self.training else torch.cat(z, 1)
        return torch.cat(z, 1), x