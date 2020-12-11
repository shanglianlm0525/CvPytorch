# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:26
# @Author : liumin
# @File : convs.py

import torch
import torch.nn as nn

from .activations import activate_layers
from .init_weights import kaiming_init, constant_init
from .norms import norm_layers


class ConvModule(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1,bias=False):
        super(ConvModule, self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,
                              padding=padding,dilation=dilation,groups=groups,bias=bias)
        if self.norm:
            self.norm_layer = norm_layers(self.norm, out_channels)
        if self.activation:
            self.activation_layer = activate_layers(self.activation)

        self.init_weights()

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.conv, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self, x):
        out = self.conv(x)
        if self.norm:
            out = self.norm_layer(out)
        if self.activation:
            out = self.activation_layer(out)
        return out


class DepthwiseConvModule(nn.Module):
    def __init__(self):
        super(DepthwiseConvModule, self).__init__()

    def forward(self, x):
        pass


class PointConvModule(nn.Module):
    def __init__(self):
        super(PointConvModule, self).__init__()

    def forward(self, x):
        pass