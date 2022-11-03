# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/24 16:17
# @Author : liumin
# @File : moat.py
import math

import torch
import torch.nn as nn

from src.models.bricks import ConvModule


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, expand_ratio=4, se_ratio=0.25, drop_connect_rate=0.2,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='GELU')):
        super(MBConvBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        self.shortcut = (stride == 1 and in_channels == out_channels)

        # expansion
        mid_channels = in_channels * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self.expand_conv = ConvModule(in_channels, mid_channels, 1, 1, 0,
                                             conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Depthwise convolution phase
        self.depthwise_conv = ConvModule(mid_channels, mid_channels, ksize, stride, (ksize - 1) // 2,
                                         groups=mid_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Squeeze and Excitation layer, if desired
        if self.se_ratio > 0:
            se_channels = max(1, int(mid_channels * se_ratio))
            self.squeeze = nn.AdaptiveAvgPool2d(1)
            self.excitation = nn.Sequential(
                nn.Conv2d(in_channels=mid_channels, out_channels=se_channels, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(in_channels=se_channels, out_channels=mid_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # project
        self.project_conv = ConvModule(mid_channels, out_channels, 1, 1, 0,
                                             conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        identity = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        if self.se_ratio > 0:
            b, c, _, _ = x.size()
            se_x = self.squeeze(x).view(b, c)
            se_x = self.excitation(se_x).view(b, c, 1, 1)
            x = x * se_x.expand_as(x)
        x = self.project_conv(x)

        if self.shortcut:
            if self.drop_connect_rate:
                x = drop_connect(x, self.drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class MOATBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, expand_ratio=4, se_ratio=0.25, drop_connect_rate=0.2,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='GELU')):
        super(MOATBlock, self).__init__()
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        self.shortcut = (stride == 1 and in_channels == out_channels)

        # expansion
        mid_channels = in_channels * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self.expand_conv = ConvModule(in_channels, mid_channels, 1, 1, 0,
                                          conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # Depthwise convolution phase
        self.depthwise_conv = ConvModule(mid_channels, mid_channels, ksize, stride, (ksize - 1) // 2,
                                         groups=mid_channels, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # project
        self.project_conv = ConvModule(mid_channels, out_channels, 1, 1, 0,
                                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None)

        self.layer_normal = nn.LayerNorm(out_channels)
        self.attention = nn.Transformer()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def forward(self, x):
        identity = x
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        if self.shortcut:
            if self.drop_connect_rate:
                x = drop_connect(x, self.drop_connect_rate, training=self.training)
            x += identity  # skip connection

        return



class MOAT(nn.Module):
    cfg = {
        'moat0': { [64, 64], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 3, 7, 2], [96, 192, 384, 768] },
        'moat1': { [64, 64], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 6, 14, 2], [96, 192, 384, 768] },
        'moat2': { [128, 128], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 6, 14, 2], [128, 256, 512, 1024] },
        'moat3': { [160, 160], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 12, 28, 2], [160, 320, 640, 1280] },
        'moat4': { [256, 256], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 12, 28, 2], [256, 512, 1024, 2048] },
        'tiny_moat0': { [32, 32], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 3, 7, 2], [32, 64, 128, 256] },
        'tiny_moat1': { [40, 40], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 3, 7, 2], [40, 80, 160, 320] },
        'tiny_moat2': { [56, 56], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 3, 7, 2], [56, 112, 224, 448] },
        'tiny_moat4': { [80, 80], ['mbconv', 'mbconv', 'moat', 'moat'], [2, 3, 7, 2], [80, 160, 320, 640] },
    }
    def __init__(self, subtype='', out_channels = [32, 48, 128, 320], out_stages=[1, 2, 3], output_stride=16,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='GELU'),
                 classifier=False, backbone_path=None, pretrained=False):
        super(MOAT, self).__init__()

        stem_size = [64, 64]
        stage_stride = [2, 2, 2, 2]
        self.stem = nn.Sequential(ConvModule(3, stem_size[0], 3, 2, 1,
                                             conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                  nn.Conv2d(stem_size[0], stem_size[1], 3, 1, 1))

        self.stage1 = nn.Sequential(

        )

    self._blocks = []
    total_num_blocks = sum(self._config.num_blocks)

    for stage_id in range(len(self._config.block_type)):
        stage_config = self._local_config(self._config, stage_id, '^stem.*')
        stage_blocks = []

        for local_block_id in range(stage_config.num_blocks):
            local_block_config = copy.deepcopy(stage_config)

            block_stride = 1
            if local_block_id == 0:
                block_stride = self._config.stage_stride[stage_id]
            local_block_config = local_block_config.replace(
                block_stride=block_stride)

            block_id = sum(self._config.num_blocks[:stage_id]) + local_block_id
            local_block_config = self._adjust_survival_rate(
                local_block_config,
                block_id, total_num_blocks)

            block_name = 'block_{:0>2d}_{:0>2d}'.format(stage_id, local_block_id)
            local_block_config.name = block_name

            if (local_block_id == stage_config.num_blocks - 1 and
                    self._config.block_type[stage_id] == 'moat' and
                    self._config.global_attention_at_end_of_moat_stage):
                local_block_config.window_size = None

            block = self._build_block(local_block_config)
            stage_blocks.append(block)

        self._blocks.append(stage_blocks)

    def forward(self, x):
        pass
