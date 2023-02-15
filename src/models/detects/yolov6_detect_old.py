# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/11 14:08
# @Author : liumin
# @File : yolov6_detect_old.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.anchors.yolov6_anchor_generator import generate_anchors
from src.models.modules.yolov6_modules import Conv

def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


class YOLOv6Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, subtype='yolov6_s', num_classes=80, in_channels=[128, 256, 512], num_anchors=1, num_layers=3, use_dfl=False, reg_max=16, depth_mul=1.0, width_mul=1.0):  # detection layer
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
        self.use_dfl = use_dfl
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))

        self.reg_max = 0 #if use_dfl is False, please set reg_max to 0
        if self.use_dfl: # for m/l
            self.reg_max = reg_max
            self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)

        # Init decouple head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
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
            nn.Conv2d(in_channels=in_channels[0],out_channels=num_classes * self.num_anchors,kernel_size=1),
            # cls_pred1
            nn.Conv2d(in_channels=in_channels[1], out_channels=num_classes * self.num_anchors,kernel_size=1),
            # cls_pred2
            nn.Conv2d(in_channels=in_channels[2],out_channels=num_classes * self.num_anchors,kernel_size=1)
        )
        self.reg_preds = nn.Sequential(
            # reg_pred0
            nn.Conv2d(in_channels=in_channels[0],out_channels=4 * (self.reg_max + self.num_anchors),kernel_size=1),
            # reg_pred1
            nn.Conv2d(in_channels=in_channels[1],out_channels=4 * (self.reg_max + self.num_anchors),kernel_size=1),
            # reg_pred2
            nn.Conv2d(in_channels=in_channels[2],out_channels=4 * (self.reg_max + self.num_anchors),kernel_size=1)
        )

        self.init_weights()

    def init_weights(self):
        prior_prob = 1e-2
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        for conv in self.reg_preds:
            b = conv.bias.view(-1,)
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        if self.use_dfl:
            self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
            self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        cls_score_list = []
        reg_dis_list = []

        for i in range(self.num_layers):
            b, _, h, w = x[i].shape
            l = h * w

            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]
            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

            cls_output = torch.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([b, self.num_classes, l]))
            reg_dis_list.append(reg_output.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
        reg_dist_list = torch.cat(reg_dis_list, axis=-1).permute(0, 2, 1)

        if self.training:
            return x, cls_score_list, reg_dist_list
        else:
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True)

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat([pred_bboxes, torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list], axis=-1)



