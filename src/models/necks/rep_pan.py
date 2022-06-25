# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/24 16:05
# @Author : liumin
# @File : rep_pan.py

import torch
from torch import nn
from src.models.modules.yolov6_modules import RepBlock, Transpose, SimConv


class RepPANNeck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolov6_s', channels=[256, 128, 128, 256, 256, 512], layers = [12, 12, 12, 12]):
        super().__init__()
        self.subtype = subtype
        assert channels is not None
        assert layers is not None

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))
        channels = list(map(lambda x: int(x * width_mul), channels))

        self.Rep_p4 = RepBlock(in_channels=channels[3] + channels[5], out_channels=channels[5], n=layers[5])
        self.Rep_p3 = RepBlock(in_channels=channels[2] + channels[6], out_channels=channels[6],n=layers[6])

        self.Rep_n3 = RepBlock(in_channels=channels[6] + channels[7], out_channels=channels[8], n=layers[7])
        self.Rep_n4 = RepBlock(in_channels=channels[5] + channels[9],out_channels=channels[10], n=layers[8])

        self.reduce_layer0 = SimConv(in_channels=channels[4], out_channels=channels[5],kernel_size=1, stride=1)
        self.upsample0 = Transpose(in_channels=channels[5], out_channels=channels[5])

        self.reduce_layer1 = SimConv(in_channels=channels[5], out_channels=channels[6], kernel_size=1, stride=1)
        self.upsample1 = Transpose(in_channels=channels[6], out_channels=channels[6])

        self.downsample2 = SimConv(in_channels=channels[6], out_channels=channels[7], kernel_size=3, stride=2)
        self.downsample1 = SimConv(in_channels=channels[8],out_channels=channels[9], kernel_size=3, stride=2)


    def forward(self, input):
        x2, x1, x0 = input

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

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs
