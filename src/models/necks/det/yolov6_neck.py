# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 9:48
# @Author : liumin
# @File : yolov6_neck.py

import torch
import torch.nn as nn
import math

from src.models.bricks import ConvModule
from src.models.modules.yolo_modules import CSPStackRep, RepBlock
from src.models.necks.det.base_det_neck import BaseDetNeck


class YOLOv5Neck(BaseDetNeck):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.6, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    csp_e_cfg = {"n": None, "t": None, "s": None, "m": float(2) / 3, "l": float(1) / 2, "x": float(1) / 2}
    def __init__(self, subtype='yolov6_s', # in_channels=[256, 512, 1024], out_channels=[128, 256, 512], num_blocks=[12, 12, 12, 12]
                 mid_channels = [128, 128, 256], depthwise=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True),**kwargs):
        super(YOLOv5Neck, self).__init__(**kwargs)
        assert isinstance(self.in_channels, list)
        self.subtype = subtype
        self.num_ins = len(self.in_channels)

        stype = subtype.split("_")[1]
        self.block = CSPStackRep if stype in ['m', 'l', 'x'] else RepBlock
        self.csp_e = self.csp_e_cfg[stype]

        depth_mul, width_mul = self.cfg[stype]
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), self.in_channels))
        self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), self.out_channels))
        self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))
        mid_channels = list(map(lambda x: int(x * width_mul), mid_channels))

        self.reduce_layer0 = ConvModule(self.in_channels[2], mid_channels[2], kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upsample0 = nn.ConvTranspose2d(mid_channels[2], mid_channels[2], kernel_size=2, stride=2, bias=True)
        self.Rep_p4 = self.block(self.in_channels[1] + mid_channels[2], mid_channels[2], n=self.num_blocks[0], e=self.csp_e)

        self.reduce_layer1 = ConvModule(mid_channels[2], mid_channels[1], kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.upsample1 = nn.ConvTranspose2d(mid_channels[1], mid_channels[1], kernel_size=2, stride=2, bias=True)
        self.Rep_p3 = self.block(self.in_channels[0] + mid_channels[1], self.out_channels[0], n=self.num_blocks[1], e=self.csp_e)

        self.downsample2 = ConvModule(self.out_channels[0], mid_channels[0], kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Rep_n3 = self.block(mid_channels[1] + mid_channels[0], self.out_channels[1], n=self.num_blocks[2], e=self.csp_e)

        self.downsample1 = ConvModule(self.out_channels[1], mid_channels[2], kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Rep_n4 = self.block(mid_channels[2] + mid_channels[2], self.out_channels[2], n=self.num_blocks[3], e=self.csp_e)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        x2, x1, x0 = x

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        return [pan_out2, pan_out1, pan_out0]


if __name__ == "__main__":
    model = YOLOv5Neck('yolov6_s', in_channels=[256, 512, 1024], out_channels=[256, 512, 1024], num_blocks=[3, 3, 3, 3])
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''