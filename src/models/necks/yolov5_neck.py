# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/2 18:03
# @Author : liumin
# @File : yolov5_neck.py
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.yolov5_modules import C3, Conv


class YOLOv5Neck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(YOLOv5Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.up_convs = nn.ModuleList()
        for i in range(self.num_ins):
            if i == 0:
                up_conv = C3(in_channels[i]*2, in_channels[i], 3, False)
            elif i< self.num_ins-1:
                up_conv = nn.Sequential(
                    C3(in_channels[i]* 2, in_channels[i], 3, False),
                    Conv(in_channels[i], in_channels[i]//2, 1, 1, 0)
                )
            else:
                up_conv = nn.Sequential(
                    C3(in_channels[i] , in_channels[i], 3, False),
                    Conv(in_channels[i], in_channels[i] // 2, 1, 1, 0)
                )
            self.up_convs.append(up_conv)

        self.down_convs1 = nn.ModuleList()
        self.down_convs2 = nn.ModuleList()
        for i in range(self.num_ins-1):
            self.down_convs1.append(Conv(in_channels[i], in_channels[i], 3, 2, 1))
            self.down_convs2.append(C3(in_channels[i]*2 , in_channels[i]*2, 3, False))

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
        outs = x
        used_backbone_levels = len(outs)
        for i in range(used_backbone_levels - 1, 0, -1):
            if i<used_backbone_levels - 1:
                prev_shape = x[i].shape[2:]
                lateral_up = F.interpolate(outs[i+1], size=prev_shape, mode='nearest')
                outs[i] = torch.cat([lateral_up, outs[i]], dim=1)
            outs[i] = self.up_convs[i](outs[i])

        for i, (down_conv1, down_conv2)  in enumerate(zip(self.down_convs1, self.down_convs2)):
            outs[i+1] = down_conv2(torch.cat([down_conv1(outs[i]), outs[i+1]], dim=1))

        return tuple(outs)


if __name__ == "__main__":
    import torch
    in_channels = [256, 512, 1024]
    out_channels = [256, 512, 1024]
    scales = [76, 38, 19]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = YOLOv5Neck(in_channels, out_channels, False).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')