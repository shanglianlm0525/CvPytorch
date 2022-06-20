# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/30 19:58
# @Author : liumin
# @File : lfd_neck.py

import torch
import torch.nn as nn
from src.models.modules.convs import ConvModule


class LFDNeck(nn.Module):
    def __init__(self, in_channels,out_channels, conv_cfg=None, norm_cfg=dict(type='BN'),activation='ReLU'):
        super(LFDNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        for i in range(self.num_ins):
            neck_conv = ConvModule(in_channels[i], out_channels, kernel_size=1, stride=1,
                                  padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, activation=activation)
            setattr(self, 'neck%d' % i, neck_conv)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert len(x) == self.num_ins

        outputs = list()
        for i in range(self._num_inputs):
            outputs.append(getattr(self, 'neck%d' % i)(x[i]))

        return outputs
