# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 9:48
# @Author : liumin
# @File : yolov6_neck.py

import torch
import torch.nn as nn
import math

from src.models.bricks import ConvModule
from src.models.modules.yolo_modules import CSPStackRep, RepBlock, BiFusion, BepC3, ConvWrapper, RepVGGBlock
from src.models.necks.det.base_det_neck import BaseDetNeck


class YOLOv6RepBiPAN(BaseDetNeck):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.6, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    csp_e_cfg = {"n": None, "t": None, "s": None, "m": float(2) / 3, "l": float(1) / 2, "x": float(1) / 2}
    def __init__(self, subtype='yolov6_s', in_channels=[128, 256, 512, 1024], mid_channels = [128, 128, 256], out_channels=[128, 256, 512], num_blocks=[12, 12, 12, 12],
                  depthwise=False, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU', inplace=True),**kwargs):
        super(YOLOv6RepBiPAN, self).__init__(subtype=subtype, cfg=self.cfg, in_channels=in_channels, mid_channels=mid_channels, out_channels=out_channels, num_blocks=num_blocks, depthwise=depthwise,
                                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        assert isinstance(self.in_channels, list)
        self.subtype = subtype
        self.num_ins = len(self.in_channels)

        stype = subtype.split("_")[1]
        self.basic_block= ConvWrapper if stype in ['l', 'x'] else RepVGGBlock
        self.block = BepC3 if stype in ['m', 'l', 'x'] else RepBlock
        self.csp_e = self.csp_e_cfg[stype]

        self.reduce_layer0 = ConvModule(self.in_channels[3], self.mid_channels[2], kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bifusion0 = BiFusion(in_channels=[self.in_channels[2], self.mid_channels[2]], out_channels=self.mid_channels[2])
        self.Rep_p4 = self.block(self.mid_channels[2], self.mid_channels[2], n=self.num_blocks[3], e=self.csp_e, block=self.basic_block)

        self.reduce_layer1 = ConvModule(self.mid_channels[2], self.mid_channels[1], kernel_size=1, stride=1, padding=0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.bifusion1 = BiFusion(in_channels=[self.mid_channels[2], self.mid_channels[1]], out_channels=self.mid_channels[1])
        self.Rep_p3 = self.block(self.mid_channels[1], self.out_channels[0], n=self.num_blocks[2], e=self.csp_e, block=self.basic_block)

        self.downsample2 = ConvModule(self.out_channels[0], self.mid_channels[0], kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Rep_n3 = self.block(self.mid_channels[1] + self.mid_channels[0], self.out_channels[1], n=self.num_blocks[1], e=self.csp_e, block=self.basic_block)

        self.downsample1 = ConvModule(self.out_channels[1], self.out_channels[1], kernel_size=3, stride=2, padding=1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.Rep_n4 = self.block(self.mid_channels[2] + self.out_channels[1], self.out_channels[2], n=self.num_blocks[0], e=self.csp_e, block=self.basic_block)

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
        x3, x2, x1, x0 = input

        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs


if __name__ == "__main__":
    model = YOLOv6RepBiPAN(subtype='yolov6_n', in_channels=[128, 256, 512, 1024], mid_channels = [128, 128, 256], out_channels=[128, 256, 512], num_blocks=[12, 12, 12, 12])
    print(model)

    '''
    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)
    '''