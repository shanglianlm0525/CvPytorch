# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 14:31
# @Author : liumin
# @File : yolox_fpn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.yolox_modules import CSPLayer, BaseConv, DWConv
from ..modules.convs import ConvModule
from ..modules.init_weights import xavier_init


class YOLOXNeck(nn.Module):

    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, subtype='yolox_s', depthwise=False, conv_cfg=None, norm_cfg=None,activation=None):
        super(YOLOXNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.subtype = subtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        if self.subtype == 'yolox_s':
            dep_mul, wid_mul = 0.33, 0.5
            self.out_channels = [32, 64, 128, 256, 512]
        elif self.subtype == 'yolox_m':
            dep_mul, wid_mul = 0.67, 0.75
            self.out_channels = [48, 96, 192, 384, 768]
        elif self.subtype == 'yolox_l':
            dep_mul, wid_mul = 1.0, 1.0
            self.out_channels = [64, 128, 256, 512, 1024]
        elif self.subtype == 'yolox_x':
            dep_mul, wid_mul = 1.33, 1.25
            self.out_channels = [80, 160, 320, 640, 1280]
        else:
            raise NotImplementedError

        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(int(in_channels[2] * wid_mul), int(in_channels[1] * wid_mul), 1, 1)
        self.C3_p4 = CSPLayer(int(2 * in_channels[1] * wid_mul),int(in_channels[1] * wid_mul),round(3 * dep_mul),False,depthwise=depthwise)  # cat

        self.reduce_conv1 = BaseConv(int(in_channels[1] * wid_mul), int(in_channels[0] * wid_mul), 1, 1)
        self.C3_p3 = CSPLayer(int(2 * in_channels[0] * wid_mul),int(in_channels[0] * wid_mul),round(3 * dep_mul),False,depthwise=depthwise)

        # bottom-up conv
        self.bu_conv2 = Conv(int(in_channels[0] * wid_mul), int(in_channels[0] * wid_mul), 3, 2)
        self.C3_n3 = CSPLayer(int(2 * in_channels[0] * wid_mul),int(in_channels[1] * wid_mul),round(3 * dep_mul),False,depthwise=depthwise)

        # bottom-up conv
        self.bu_conv1 = Conv(int(in_channels[1] * wid_mul), int(in_channels[1] * wid_mul), 3, 2)
        self.C3_n4 = CSPLayer(int(2 * in_channels[1] * wid_mul),int(in_channels[2] * wid_mul),round(3 * dep_mul),False,depthwise=depthwise)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, x):
        assert len(x) == len(self.in_channels)

        fpn_out0 = self.lateral_conv0(x[2])  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x[1]], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x[0]], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outs = (pan_out2, pan_out1, pan_out0)
        return outs


if __name__ == '__main__':
    import torch
    in_channels = [2, 3, 5]
    scales = [340, 170, 84]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = YOLOXNeck(in_channels, 11, False).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')