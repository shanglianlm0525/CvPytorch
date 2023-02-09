# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/2/7 10:51
# @Author : liumin
# @File : yolov6_effidehead.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.bricks import ConvModule
from src.models.heads.det.base_yolo_head import BaseYOLOHead


class YOLOv6Effidehead(BaseYOLOHead):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.6, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    csp_e_cfg = {"n": None, "t": None, "s": None, "m": float(2) / 3, "l": float(1) / 2, "x": float(1) / 2}
    def __init__(self, subtype='yolov6_s', num_classes=80, in_channels=[128, 256, 512], num_anchors=1, reg_max=16,
                 stacked_convs=2, depthwise=False, use_dfl=False, fuse_ab=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'), **kwargs):
        super(YOLOv6Effidehead, self).__init__(subtype=subtype, cfg=self.cfg, num_classes=num_classes, in_channels=in_channels, stacked_convs=stacked_convs, depthwise=depthwise,
                                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.subtype = subtype
        stype = subtype.split("_")[1]
        self.fuse_ab = fuse_ab
        self.num_anchors = 3 if self.fuse_ab else 1
        self.reg_max = 16 if stype in ['m', 'l', 'x'] else 0
        self.stacked_convs = stacked_convs
        self.use_dfl = use_dfl

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)

        # Efficient decoupled head layers
        self.stems = nn.Sequential(
            # stem0
            ConvModule(in_channels=self.in_channels[0], out_channels=self.in_channels[0], kernel_size=1, stride=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # stem1
            ConvModule(in_channels=self.in_channels[1], out_channels=self.in_channels[1], kernel_size=1, stride=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # stem2
            ConvModule(in_channels=self.in_channels[2], out_channels=self.in_channels[2], kernel_size=1, stride=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        )
        self.cls_convs = nn.Sequential(
            # cls_conv0
            ConvModule(in_channels=self.in_channels[0], out_channels=self.in_channels[0], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # cls_conv1
            ConvModule(in_channels=self.in_channels[1], out_channels=self.in_channels[1], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # cls_conv2
            ConvModule(in_channels=self.in_channels[2], out_channels=self.in_channels[2], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        )
        self.reg_convs = nn.Sequential(
            # reg_conv0
            ConvModule(in_channels=self.in_channels[0], out_channels=self.in_channels[0], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # reg_conv1
            ConvModule(in_channels=self.in_channels[1], out_channels=self.in_channels[1], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg),
            # reg_conv2
            ConvModule(in_channels=self.in_channels[2], out_channels=self.in_channels[2], kernel_size=3, stride=1, padding=1, conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
        )
        self.cls_preds = nn.Sequential(
            # cls_pred0
            nn.Conv2d(in_channels=self.in_channels[0], out_channels=self.num_classes * self.num_anchors, kernel_size=1),
            # cls_pred1
            nn.Conv2d(in_channels=self.in_channels[1], out_channels=self.num_classes * self.num_anchors, kernel_size=1),
            # cls_pred2
            nn.Conv2d(in_channels=self.in_channels[2], out_channels=self.num_classes * self.num_anchors, kernel_size=1)
        )
        self.reg_preds = nn.Sequential(
            # reg_pred0
            nn.Conv2d(in_channels=self.in_channels[0], out_channels=4 * (self.num_anchors + self.reg_max), kernel_size=1),
            # reg_pred1
            nn.Conv2d(in_channels=self.in_channels[1], out_channels=4 * (self.num_anchors + self.reg_max), kernel_size=1),
            # reg_pred2
            nn.Conv2d(in_channels=self.in_channels[2], out_channels=4 * (self.num_anchors + self.reg_max), kernel_size=1)
        )

        self.init_weights()


    def init_weights(self, prior_prob=1e-2):
        prior_prob = 1e-2
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
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
                reg_dist_list.append(reg_output.reshape([b, 4, l]))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)

            return x, cls_score_list, reg_dist_list


if __name__ == "__main__":
    model = YOLOv6Effidehead(subtype='yolov6_n', num_classes = 80, in_channels= [128, 256, 512],
                      norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),act_cfg=dict(type='SiLU'))
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''
