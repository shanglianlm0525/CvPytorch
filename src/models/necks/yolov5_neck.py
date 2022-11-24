# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/2 18:03
# @Author : liumin
# @File : yolov5_neck.py

import torch
import torch.nn as nn
from src.models.modules.yolo11_modules import UpsamplingModule, DownsamplingModule


class YOLOv5Neck(nn.Module):
    def __init__(self, in_channels, out_channels, depth_mul=1.0, width_mul=1.0):
        super(YOLOv5Neck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
        self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), out_channels))

        layers = [3, 3, 3, 3]
        self.layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))

        self.up_1 = UpsamplingModule(self.in_channels[2], self.in_channels[1], self.layers[0])
        self.up_2 = UpsamplingModule(self.in_channels[1], self.out_channels[0], self.layers[1])

        self.down_1 = DownsamplingModule(self.in_channels[0], self.in_channels[1], self.layers[2])
        self.down_2 = DownsamplingModule(self.in_channels[1], self.in_channels[2], self.layers[3])

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
        x3, x4, x5 = x
        # up
        x4_up, x4_t = self.up_1(x5, x4)
        x3_up, x3_t = self.up_2(x4_up, x3)
        # down
        x4_down = self.down_1(x3_up, x3_t)
        x5_down = self.down_2(x4_down, x4_t)
        return [x3_up, x4_down, x5_down]


if __name__ == "__main__":
    import torch
    in_channels = [256, 512, 1024]
    out_channels = [256, 512, 1024]
    scales = [80, 40, 20]
    depth_mul = 1.33
    width_mul = 1.25
    in_channels1 = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels1, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    model = YOLOv5Neck(in_channels, out_channels, depth_mul=depth_mul, width_mul=width_mul).eval()
    print(model)
    outputs = model(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')

