# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/6 14:29
# @Author : liumin
# @File : tood_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.heads import YOLOXHead
from src.models.modules.convs import ConvModule


"""
    TOOD: Task-aligned One-stage Object Detection
    https://arxiv.org/pdf/2108.07755.pdf
"""

class TOODHead(YOLOXHead):
    def __init__(self, num_classes=80, subtype='pai_yolox_s', in_channels=[256, 512, 1024], out_channels=[256, 512, 1024], feat_channels=256, stacked_convs=3, strides = [8, 16, 32],
                 la_down_rate=32, norm_cfg = dict(type='GN', num_groups=32, requires_grad=True), depth_mul=1.0, width_mul=1.0):
        super(TOODHead, self).__init__(num_classes, subtype, in_channels, strides, depth_mul, width_mul)
        self.num_classes = num_classes  # number of classes
        self.subtype = subtype
        self.strides = strides
        self.stacked_convs = stacked_convs
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))
        out_channels = list(map(lambda x: int(x * width_mul), out_channels))
        self.feat_channels = int(feat_channels * width_mul)

        self.cls_decomps = nn.ModuleList()
        self.reg_decomps = nn.ModuleList()
        self.inter_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            self.cls_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate, norm_cfg = norm_cfg))
            self.reg_decomps.append(
                TaskDecomposition(self.feat_channels, self.stacked_convs,
                                  self.stacked_convs * la_down_rate, norm_cfg = norm_cfg))

        for i in range(self.stacked_convs):
            self.inter_convs.append(ConvModule(self.feat_channels, self.feat_channels, 3, 1, 1,norm_cfg=norm_cfg))

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, x):
        outputs = []
        for k, (cls_decomp, reg_decomp, cls_conv, reg_conv, xx) in \
                enumerate(zip(self.cls_decomps, self.reg_decomps, self.cls_convs, self.reg_convs, x)):
            xx = self.stems[k](xx)

            # inter feature maps
            inter_feats = []
            for inter_conv in self.inter_convs:
                xx = inter_conv(xx)
                inter_feats.append(xx)
            feat = torch.cat(inter_feats, 1)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_x = cls_decomp(feat, avg_feat)
            reg_x = reg_decomp(feat, avg_feat)

            # classify
            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            # regress, object, (reid)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self, feat_channels, stacked_convs=6, la_down_rate=8, conv_cfg=None, norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channels // la_down_rate,
                self.stacked_convs, 1,padding=0),
            nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,self.feat_channels, 1, 1, 0,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.001)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        m = self.reduction_conv.conv
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0, 0.01)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
                1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.act(feat)
        return feat