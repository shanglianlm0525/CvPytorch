# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/8 16:13
# @Author : liumin
# @File : rfcr.py

import torch
import torch.nn as nn


class RFCR(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(RFCR, self).__init__()

        self.collection1 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4),
                                         nn.Conv2d(in_channels[0], out_channel, 1, 1, 0))
        self.collection2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                         nn.Conv2d(in_channels[1], out_channel, 1, 1, 0))
        self.collection3 = nn.Conv2d(in_channels[2], out_channel, 1, 1, 0)
        self.collection4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                         nn.Conv2d(in_channels[3], out_channel, 1, 1, 0))

        # Weight
        self.w1 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.w3 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)
        self.w4 = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=True)

        self.fuse = nn.Conv2d(out_channel, out_channel, 5, 1, 2)

        self.redistribute2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.redistribute4 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x1, x2, x3, x4 = x
        cat_x = self.w1 * self.collection1(x1) + self.w2 * self.collection2(x2) + self.w3 * self.collection3(x3) + self.w4 * self.collection4(x4)
        cat_x = self.fuse(cat_x)
        return [torch.cat([cat_x, self.redistribute2(x2)]), torch.cat([cat_x, x3]), torch.cat([cat_x, self.redistribute4(x4)])]


if __name__ == '__main__':
    import torch
    in_channels = [2, 3, 5, 7]
    scales = [160, 80, 40, 20]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    model = RFCR(in_channels, 11)
    outputs = model(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')
