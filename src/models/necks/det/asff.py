# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 10:51
# @Author : liumin
# @File : asff.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import ConvModule

"""
    Learning Spatial Fusion for Single-Shot Object Detection
    https://arxiv.org/pdf/1911.09516.pdf
"""

class ASFF(nn.Module):
    def __init__(self, level=0, in_channels=[256, 512, 1024], asff_channel=2, expand_kernel=3,
                 norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True)):
        super(ASFF, self).__init__()
        self.in_channels= in_channels
        self.asff_channel = asff_channel
        self.level = level

        self.mid_channels = self.in_channels[self.level]

        if self.level == 0:
            self.stride_level_1 = ConvModule(self.in_channels[1], self.mid_channels, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.stride_level_2 = ConvModule(self.in_channels[2], self.mid_channels, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        elif self.level == 1:
            self.compress_level_0 = ConvModule(self.in_channels[0], self.mid_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.stride_level_2 = ConvModule(self.in_channels[2], self.mid_channels, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        elif self.level == 2:
            self.compress_level_0 = ConvModule(self.in_channels[0], self.mid_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
            self.compress_level_1 = ConvModule(self.in_channels[1], self.mid_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        else:
            raise ValueError('Invalid level {}'.format(self.level))

        # add expand layer
        self.expand = ConvModule(self.mid_channels, self.mid_channels, expand_kernel, 1, padding=expand_kernel//2, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.weight_level_0 = ConvModule(self.mid_channels, self.asff_channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.weight_level_1 = ConvModule(self.mid_channels, self.asff_channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.weight_level_2 = ConvModule(self.mid_channels, self.asff_channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.weight_levels = ConvModule(self.asff_channel * 3, 3, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)


    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left,patch_bot_left,patch_top_right,patch_bot_right), dim=1,)
        return x


    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/4,h*2,w*2]
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2


    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(
                level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(
                level_0_compressed, scale_factor=4, mode='nearest')
            x_level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(
                x_level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :,:] + \
                            level_2_resized * levels_weight[:,2:, :, :]
        out = self.expand(fused_out_reduced)
        return out