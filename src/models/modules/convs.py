# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/10 15:26
# @Author : liumin
# @File : convs.py

import torch
import torch.nn as nn

from .activations import build_activate_layer
from .init_weights import kaiming_init, constant_init
from .norms import build_norm_layer


class ConvModule(nn.Module):
    '''A conv block that bundles conv/norm/activation layers.'''
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1,bias=False,norm=None,activation=None):
        super(ConvModule, self).__init__()
        self.norm = norm
        self.activation = activation

        self.conv_layer = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,
                              padding=padding,dilation=dilation,groups=groups,bias=bias)
        if self.norm is not None:
            self.norm_layer = build_norm_layer(self.norm, out_channels)
        if self.activation is not None:
            self.activation_layer = build_activate_layer(self.activation)

        self.init_weights()

    def init_weights(self):
        if self.activation == 'LeakyReLU':
            nonlinearity = 'leaky_relu'
        else:
            nonlinearity = 'relu'
        kaiming_init(self.conv_layer, nonlinearity=nonlinearity)
        if self.norm:
            constant_init(self.norm_layer, 1, bias=0)

    def forward(self, x):
        out = self.conv_layer(x)
        if self.norm is not None:
            out = self.norm_layer(out)
        if self.activation is not None:
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