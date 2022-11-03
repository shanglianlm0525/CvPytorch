# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 9:48
# @Author : liumin
# @File : yolov5_neck.py

import torch
import torch.nn as nn
import math

from src.models.modules.yolo_modules import UpsamplingModule, DownsamplingModule
from src.models.necks.det.base_det_neck import BaseDetNeck


class YOLOv5Neck(BaseDetNeck):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolov5_s', # in_channels=[256, 512, 1024], out_channels=256, num_blocks=[3, 3, 3, 3]
                 depthwise=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True),**kwargs):
        super(YOLOv5Neck, self).__init__(**kwargs)
        assert isinstance(self.in_channels, list)
        self.subtype = subtype
        self.num_ins = len(self.in_channels)

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), self.out_channels))
        self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))

        self.up_1 = UpsamplingModule(self.in_channels[2], self.in_channels[1], self.num_blocks[0], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up_2 = UpsamplingModule(self.in_channels[1], self.out_channels[0], self.num_blocks[1], norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.down_1 = DownsamplingModule(self.in_channels[0], self.in_channels[1], self.num_blocks[2], norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.down_2 = DownsamplingModule(self.in_channels[1], self.in_channels[2], self.num_blocks[3], norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        x3, x4, x5 = x
        # up
        x4_up, x4_t = self.up_1(x5, x4)
        x3_up, x3_t = self.up_2(x4_up, x3)
        # down
        x4_down = self.down_1(x3_up, x3_t)
        x5_down = self.down_2(x4_down, x4_t)
        return [x3_up, x4_down, x5_down]


if __name__ == "__main__":
    model = YOLOv5Neck('yolov5_s', in_channels=[256, 512, 1024], out_channels=[256, 512, 1024], num_blocks=[3, 3, 3, 3])
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''