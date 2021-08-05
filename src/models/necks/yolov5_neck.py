# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/2 18:03
# @Author : liumin
# @File : yolov5_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# from ..modules.yolov5_modules import C3, Conv
from src.models.modules.yolov5_modules import Conv, C3


class YOLOv5Neck(nn.Module):
    def __init__(self, in_channels, out_channels, layers=[3, 3, 3], depth_mul=1.0, width_mul=1.0):
        super(YOLOv5Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.width_mul = width_mul
        self.depth_mul = depth_mul
        self.in_channels = list(map(lambda x: max(round(x * self.width_mul), 1), in_channels))
        self.out_channels = list(map(lambda x: max(round(x * self.width_mul), 1), out_channels))
        self.num_ins = len(in_channels)
        self.layers = list(map(lambda x: max(round(x * self.depth_mul), 1), layers))

        self.up_convs = nn.ModuleList()
        for i in range(self.num_ins-1, 0, -1):
            if i == self.num_ins-1:
                up_conv = nn.Sequential(
                    C3(self.in_channels[i], self.in_channels[i], self.layers[i], False),
                    Conv(self.in_channels[i], self.in_channels[i-1], 1, 1, 0)
                )
                self.up_convs.append(up_conv)
            else:
                up_conv = nn.Sequential(
                    C3(self.in_channels[i] * 2, self.in_channels[i], self.layers[i], False),
                    Conv(self.in_channels[i], self.in_channels[i-1], 1, 1, 0)
                )
                self.up_convs.append(up_conv)


        self.down_convs1 = nn.ModuleList()
        self.down_convs2 = nn.ModuleList()
        for i in range(self.num_ins):
            if i == 0:
                self.down_convs1.append(C3(self.in_channels[i] * 2, self.out_channels[i], 3, False))
            else:
                self.down_convs1.append(C3(self.in_channels[i], self.out_channels[i], 3, False))
            if i < self.num_ins-1:
                self.down_convs2.append(Conv(self.out_channels[i], self.in_channels[i], 3, 2, 1))

        self.init_weights()


    def init_weights(self):
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
        assert len(x) == len(self.in_channels)
        outs = list(x)
        used_backbone_levels = len(outs)
        for i in range(used_backbone_levels - 1, 0, -1):
            outs[i] = self.up_convs[used_backbone_levels - 1 - i](outs[i])
            lateral_up = F.interpolate(outs[i], size=x[i-1].shape[2:], mode='nearest')
            outs[i-1] = torch.cat([lateral_up, outs[i-1]], dim=1)

        for i in range(used_backbone_levels):
            if i < used_backbone_levels-1:
                outs[i] = self.down_convs1[i](outs[i])
                outs[i+1] = torch.cat([self.down_convs2[i](outs[i]), outs[i+1]], dim=1)
            else:
                outs[i] = self.down_convs1[i](outs[i])

        return tuple(outs)


if __name__ == "__main__":
    import torch
    in_channels = [256, 512, 1024]
    out_channels = [256, 512, 1024]
    scales = [80, 40, 20]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = YOLOv5Neck(in_channels, out_channels, depth_mul = 1, width_mul = 1).eval()
    print(self)
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')

