# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/11 13:14
# @Author : liumin
# @File : yolov7_neck.py

import torch
import torch.nn as nn

from src.models.modules.yolov7_modules import UpSampling, FeatureFusion, DownB, SPPCSPC


class YOLOv7Neck(nn.Module):
    def __init__(self, in_channels=[512, 1024, 1024], out_channels=[128, 256, 512], depth_mul=1.0, width_mul=1.0):
        super(YOLOv7Neck, self).__init__()
        assert isinstance(in_channels, list)
        in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
        out_channels = list(map(lambda x: max(round(x * width_mul), 1), out_channels))

        self.spp = SPPCSPC(in_channels[2], in_channels[0])
        self.up1_1 = UpSampling(in_channels[0], in_channels[1], out_channels[1])
        self.featurefusion1_1 = FeatureFusion(out_channels[1]*2, out_channels[1])

        self.up1_2 = UpSampling(out_channels[1], in_channels[0], out_channels[0])
        self.featurefusion1_2 = FeatureFusion(out_channels[0]*2, out_channels[0])

        self.down2_1 = DownB(out_channels[0], out_channels[0])
        self.featurefusion2_1 = FeatureFusion(out_channels[1]*2, out_channels[1])

        self.down2_2 = DownB(out_channels[1], out_channels[1])
        self.featurefusion2_2 = FeatureFusion(out_channels[2]*2, out_channels[2])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def forward(self, x):
        x3, x4, x5 = x
        x5 = self.spp(x5)
        # up
        x4_up = self.featurefusion1_1(self.up1_1(x5, x4))
        x3_up = self.featurefusion1_2(self.up1_2(x4_up, x3))
        # down
        x4_down = self.featurefusion2_1(self.down2_1(x3_up, x4_up))
        x5_down = self.featurefusion2_2(self.down2_2(x4_down, x5))
        return [x3_up, x4_down, x5_down]


if __name__ == '__main__':
    import torch
    in_channels = [512, 1024, 512]
    out_channels = [128, 256, 512]
    scales = [80, 40, 20]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = YOLOv7Neck(in_channels, out_channels).eval()
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')