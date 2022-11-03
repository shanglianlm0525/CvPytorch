# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 9:48
# @Author : liumin
# @File : yolox_pai_neck.py


import torch
import torch.nn as nn
import math

from src.models.necks.det.asff import ASFF
from src.models.necks.det.base_det_neck import BaseDetNeck


class YOLOXPAINeck(BaseDetNeck):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolox_pai_s', asff_channel=2, expand_kernel=3, # in_channels=[256, 512, 1024], out_channels=[256, 512, 1024], num_blocks=[3, 3, 3, 3],
                 depthwise=False, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='Swish'),**kwargs):
        super(YOLOXPAINeck, self).__init__(**kwargs)
        assert isinstance(self.in_channels, list)
        self.subtype = subtype
        self.num_ins = len(self.in_channels)

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        # self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), self.out_channels))
        self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))

        self.asff_1 = ASFF(level=0, in_channels=self.in_channels, asff_channel=asff_channel, expand_kernel=expand_kernel, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.asff_2 = ASFF(level=1, in_channels=self.in_channels, asff_channel=asff_channel, expand_kernel=expand_kernel, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.asff_3 = ASFF(level=2, in_channels=self.in_channels, asff_channel=asff_channel, expand_kernel=expand_kernel, norm_cfg=norm_cfg, act_cfg=act_cfg)

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
        outs = super(YOLOXPAINeck, self).forward(x)

        asff_out0 = self.asff_1(outs)
        asff_out1 = self.asff_2(outs)
        asff_out2 = self.asff_3(outs)
        return [asff_out2, asff_out1, asff_out0]


if __name__ == "__main__":
    model = YOLOXPAINeck('yolox_pai_s')
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''