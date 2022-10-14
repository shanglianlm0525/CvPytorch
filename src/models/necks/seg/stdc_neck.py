# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 20:38
# @Author : liumin
# @File : stdc_neck.py

import torch
from torch import nn
import torch.nn.functional as F

from src.base.base_module import BaseModule
from src.models.bricks import ConvModule
from src.models.necks.seg.base_seg_neck import BaseSegNeck


class AttentionRefinementModule(BaseModule):
    """Attention Refinement Module (ARM) to refine the features of each stage.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
    Returns:
        x_out (torch.Tensor): Feature map of Attention Refinement Module.
    """

    def __init__(self,
                 in_channels,
                 out_channel,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(AttentionRefinementModule, self).__init__(init_cfg=init_cfg)
        self.conv_layer = ConvModule(
            in_channels=in_channels,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.atten_conv_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=1,
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None), nn.Sigmoid())

    def forward(self, x):
        x = self.conv_layer(x)
        x_atten = self.atten_conv_layer(x)
        x_out = x * x_atten
        return x_out


class FeatureFusionModule(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super(FeatureFusionModule, self).__init__(init_cfg=init_cfg)
        channels = out_channels // scale_factor
        self.conv0 = ConvModule(
            in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvModule(
                out_channels,
                channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=act_cfg),
            ConvModule(
                channels,
                out_channels,
                1,
                norm_cfg=None,
                bias=False,
                act_cfg=None), nn.Sigmoid())

    def forward(self, spatial_inputs, context_inputs):
        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.conv0(inputs)
        attn = self.attention(x)
        x_attn = x * attn
        return x_attn + x


class STDCNeck(BaseSegNeck):
    def __init__(self, in_channels=[256, 512, 1024], out_channels=256, aux_out_channels=128, norm_cfg=dict(type='BN'), **kwargs):
        super(STDCNeck, self).__init__()
        self.arms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for in_channel in in_channels[1:]:
            self.arms.append(AttentionRefinementModule(in_channel, aux_out_channels))
            self.convs.append(ConvModule(aux_out_channels, aux_out_channels, 3, padding=1, norm_cfg=self.norm_cfg))

        self.conv_avg = ConvModule(in_channels[-1], aux_out_channels, 1, norm_cfg=self.norm_cfg)

        self.ffm = FeatureFusionModule(in_channels=in_channels[0]+aux_out_channels, out_channels=out_channels)

    def forward(self, x):
        # feat32_avg
        avg = F.adaptive_avg_pool2d(x[-1], 1)
        avg_feat = self.conv_avg(avg)
        feature_up = F.interpolate(avg_feat, x[-1].shape[2:], mode='nearest')

        # feat32/feat16
        arms_out = []
        for i in range(len(self.arms)-1, -1, -1):
            x_arm = self.arms[i](x[i+1]) + feature_up
            feature_up = F.interpolate(x_arm, x[i].shape[2:], mode='nearest')
            feature_up = self.convs[i](feature_up)
            arms_out.append(feature_up)

        feat_fuse = self.ffm(x[0], arms_out[1])
        return feat_fuse, [x[0]] + arms_out


if __name__ == '__main__':
    model = STDCNeck(in_channels=[256, 512, 1024], aux_out_channels=128, out_channels=256)
    print(model)

    input1 = torch.randn(2, 256, 80, 80)
    input2 = torch.randn(2, 512, 40, 40)
    input3 = torch.randn(2, 1024, 20, 20)
    input= [input1, input2, input3]
    out, aux_out = model(input)

    print(out.shape)
    for o in aux_out:
        print(o.shape)