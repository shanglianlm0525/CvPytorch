# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 13:48
# @Author : liumin
# @File : yolox_head.py

import torch
import torch.nn as nn
import numpy as np

from src.models.modules.convs import ConvModule


class YOLOXHead(nn.Module):
    def __init__(self, num_classes=80, subtype='yolox_s', in_channels=[256, 512, 1024], strides = [8, 16, 32], depth_mul=1.0, width_mul=1.0):
        super(YOLOXHead, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.subtype = subtype
        self.strides = strides
        self.n_anchors = 1

        in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
        in_places = int(256 * width_mul)

        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(
                ConvModule(in_channels=in_channels[i], out_channels=in_places, kernel_size=1, stride=1, padding=1, activation="SiLU")
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[ConvModule(in_channels=in_places, out_channels=in_places, kernel_size=3, stride=1, padding=1, activation="SiLU"),
                    ConvModule(in_channels=in_places, out_channels=in_places, kernel_size=3, stride=1, padding=1, activation="SiLU"),]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[ConvModule(in_channels=in_places, out_channels=in_places, kernel_size=3, stride=1, padding=1, activation="SiLU"),
                    ConvModule(in_channels=in_places, out_channels=in_places, kernel_size=3, stride=1, padding=1, activation="SiLU"),]
                )
            )
            self.cls_preds.append(nn.Conv2d(in_places, self.n_anchors * self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(in_places,4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(in_places, self.n_anchors * 1, 1, 1, 0))

        self.init_weights()


    def init_weights(self, prior_prob=1e-2):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-np.math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, feats):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, feats)):
            x = self.stems[k](x)

            # classify
            cls_feat = cls_conv(x)
            cls_output = self.cls_preds[k](cls_feat)

            # regress, object, (reid)
            reg_feat = reg_conv(x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs


