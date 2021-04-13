# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 14:57
# @Author : liumin
# @File : yolov3_neck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.modules.convs import ConvModule


class YOLOv3Neck(nn.Module):
    '''modified from MMDetection'''

    def __init__(self, in_channels=[256, 512, 1024],
                 out_channels=[128, 256, 512]):
        super(YOLOv3Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels

        for i, (in_c, out_c) in enumerate(
                zip(reversed(self.in_channels), reversed(self.out_channels))):
            if i > 0:
                self.add_module(f'conv{i}', ConvModule(in_c, out_c, kernel_size=1, stride=1,
                                           padding=0, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'lateral_conv{i}',
                            self.make_lateral_conv(in_c + out_c if i>0 else in_c, out_c))

    def make_lateral_conv(self, in_c, out_c):
        out_c2 = out_c * 2
        return nn.Sequential(
            ConvModule(in_c, out_c, kernel_size=1, stride=1,
                       padding=0, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'),
            ConvModule(out_c, out_c2, kernel_size=3, stride=1,
                       padding=1, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'),
            ConvModule(out_c2, out_c, kernel_size=1, stride=1,
                       padding=0, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'),
            ConvModule(out_c, out_c2, kernel_size=3, stride=1,
                       padding=1, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'),
            ConvModule(out_c2, out_c, kernel_size=1, stride=1,
                       padding=0, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'),
        )

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        outs = []
        # processed from bottom (high-lvl) to top (low-lvl)
        for i, x in enumerate(reversed(feats)):
            if i > 0:
                conv = getattr(self, f'conv{i}')
                lateral = conv(out)
                # Cat with low-lvl feats
                lateral = F.interpolate(lateral, scale_factor=2)
                lateral = torch.cat((lateral, x), 1)
            else:
                lateral = x
            lateral_conv = getattr(self, f'lateral_conv{i}')
            out = lateral_conv(lateral)
            outs.append(out)
        return tuple(outs)


if __name__ == '__main__':
    in_channels = [256, 512, 1024]
    scales = [32, 16, 8]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    model = YOLOv3Neck(in_channels, [128, 256, 512])
    print(model)
    outputs = model(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')
