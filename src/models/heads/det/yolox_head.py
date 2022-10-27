# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 14:29
# @Author : liumin
# @File : yolox_head.py
import math

import torch
import torch.nn as nn
import numpy as np

from src.models.bricks import ConvModule, DepthwiseSeparableConvModule
from src.models.heads.det.base_det_head import BaseDetHead


class YOLOXHead(BaseDetHead):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolox_s', # 'num_classes': 80, 'in_channels': 256, 'channels': 256,
                 stacked_convs=2, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='Swish'), **kwargs):
        super(YOLOXHead, self).__init__(**kwargs)
        self.subtype = subtype
        self.stacked_convs = stacked_convs

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = max(round(self.in_channels * width_mul), 1)
        self.channels = max(round(self.channels * width_mul), 1)

        conv = DepthwiseSeparableConvModule if self.depthwise else ConvModule

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()

        for _ in self.strides:
            # self.stems.append(conv(in_channels=self.in_channels, out_channels=self.channels, kernel_size=1, stride=1, padding=1,
            #               conv_cfg=self.conv_cfg,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg))

            stacked_cls_convs = []
            stacked_reg_convs = []
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.channels
                stacked_cls_convs.append(conv(chn, self.channels, 3, 1, 1,
                                              conv_cfg=conv_cfg,norm_cfg=norm_cfg,act_cfg=act_cfg))
                stacked_reg_convs.append(conv(chn, self.channels, 3, 1, 1,
                                              conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.cls_convs.append(nn.Sequential(*stacked_cls_convs))
            self.reg_convs.append(nn.Sequential(*stacked_reg_convs))

            self.cls_preds.append(nn.Conv2d(self.channels, self.num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(self.channels, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(self.channels, 1, 1, 1, 0))

        self.init_weights()


    def init_weights(self, prior_prob=1e-2):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        # Use prior in model initialization to improve stability
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        for conv in self.cls_preds:
            conv.bias.data.fill_(bias_init)

        for conv in self.obj_preds:
            conv.bias.data.fill_(bias_init)


    def forward(self, x):
        outputs = []
        for k, (cls_conv, reg_conv, xx) in enumerate(zip(self.cls_convs, self.reg_convs, x)):
            # xx = self.stems[k](xx)

            # classify
            cls_feat = cls_conv(xx)
            cls_output = self.cls_preds[k](cls_feat)

            # regress, object, (reid)
            reg_feat = reg_conv(xx)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)

        return outputs


if __name__ == "__main__":
    model = YOLOXHead('yolox_s', num_classes = 80, in_channels= 256, channels=256, stacked_convs=2,
                      norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),act_cfg=dict(type='Swish'))
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''
    import vpi