# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/6 15:30
# @Author : liumin
# @File : stdc_head.py

import torch
from torch import nn
import torch.nn.functional as F

from src.models.modules.convs import ConvModule



class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvModule(in_channel, out_channel, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')
        self.conv_atten = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channel)

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channel, out_channel, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvModule(in_channel, out_channel, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')
        self.conv1 = nn.Conv2d(out_channel, out_channel // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if ly.bias is not None:
                    nn.init.constant_(ly.bias, 0)


class StdcHead(nn.Module):
    def __init__(self, num_classes, in_channels, mid_channel = 128):
        super(StdcHead, self).__init__()
        # 256, 512, 1024

        self.conv_avg = ConvModule(in_channels[2], 128, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU')

        self.arm32 = AttentionRefinementModule(in_channels[2], 128)
        self.conv_head32 = ConvModule(128, 128, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')

        self.arm16 = AttentionRefinementModule(in_channels[1], 128)
        self.conv_head16 = ConvModule(128, 128, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')

        self.ffm = FeatureFusionModule(in_channels[0] + mid_channel, 256)

        self.conv_out8 = nn.Sequential(
                ConvModule(in_channels[0], 256, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU'),
                nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
            )
        self.conv_out16 = nn.Sequential(
                ConvModule(mid_channel, 64, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU'),
                nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
            )
        self.conv_out32 = nn.Sequential(
                ConvModule(mid_channel, 64, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU'),
                nn.Conv2d(64, num_classes, kernel_size=1, bias=False)
            )

        self.conv_out_sp8 = nn.Sequential(
                ConvModule(in_channels[0], 64, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU'),
                nn.Conv2d(64, 1, kernel_size=1, bias=False)
            )

        self._init_weight()

    def forward(self, x):
        feat8, feat16, feat32 = x

        # feat32_avg
        feat32_avg = F.adaptive_avg_pool2d(feat32, 1)
        feat32_avg = self.conv_avg(feat32_avg)
        feat32_avg_up = F.interpolate(feat32_avg, feat32.size()[2:], mode='nearest')

        # feat32
        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + feat32_avg_up
        feat32_up = F.interpolate(feat32_sum, feat16.size()[2:], mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        # feat16
        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, feat8.size()[2:], mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        # ffm
        feat8_fuse = self.ffm(feat8, feat16_up)

        # out
        feat_out8 = self.conv_out8(feat8_fuse)
        feat_out16 = self.conv_out16(feat16_up)
        feat_out32 = self.conv_out32(feat32_up)

        feat_out_sp8 = self.conv_out_sp8(feat8)

        return feat_out8, feat_out16, feat_out32, feat_out_sp8

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)