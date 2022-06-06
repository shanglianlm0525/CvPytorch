# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/2 19:49
# @Author : liumin
# @File : ppliteseg_head.py

import torch
from torch import nn
import torch.nn.functional as F


class SimplePyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, mid_channel, sppm_channel, sizes=[1, 2, 4]):
        super(SimplePyramidPoolingModule, self).__init__()

        self.stages = nn.ModuleList([
            nn.Sequential(nn.AdaptiveAvgPool2d(output_size=size),
                          nn.Conv2d(in_channels, mid_channel, 1, 1, 0, bias=False),
                          nn.BatchNorm2d(mid_channel),
                          nn.ReLU(inplace=True))
            for size in sizes
        ])

        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channel, sppm_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(sppm_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        xshape = x.shape[2:]
        out = None
        for stage in self.stages:
            feat = stage(x)
            feat = F.interpolate(feat, xshape, mode='bilinear', align_corners=True)
            if out is None:
                out = feat
            else:
                out += feat
        out = self.conv_out(out)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialAttentionModule, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        mean_x = torch.mean(x, dim=1, keepdim=True)
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        mean_y = torch.mean(y, dim=1, keepdim=True)
        max_y = torch.max(y, dim=1, keepdim=True)[0]
        feats = torch.cat([mean_x, max_x, mean_y, max_y], dim=1)
        atten = self.fuse(feats)

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttentionModule, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fuse = nn.Sequential(
            nn.Conv2d(4, 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        max_x = self.maxpool(x)
        avg_x = self.avgpool(x)
        max_y = self.maxpool(y)
        avg_y = self.avgpool(y)
        feats = torch.cat([avg_x, max_x, avg_y, max_y], dim=1)
        atten = self.fuse(feats)

        out = x * atten + y * (1 - atten)
        out = self.conv_out(out)
        return out


class UnifiedAttentionFusionModule(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, use_sa=True):
        super(UnifiedAttentionFusionModule, self).__init__()
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        if use_sa:
            self.attention = SpatialAttentionModule(mid_channels, mid_channels)
        else:
            self.attention = ChannelAttentionModule(mid_channels, mid_channels)

        self.conv_out = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, high_feat):
        low_feat = self.proj_conv(low_feat)
        high_up = F.interpolate(high_feat, size=low_feat.size()[2:], mode='bilinear', align_corners=True)
        out = self.attention(low_feat, high_up)
        out = self.conv_out(out)
        return out


class PPLiteSegHead(nn.Module):
    def __init__(self, num_classes, in_channels, out_channels, sppm_channel, sizes):
        super(PPLiteSegHead, self).__init__()
        assert isinstance(in_channels, list) and len(in_channels) == 3

        self.sppm = SimplePyramidPoolingModule(in_channels[2], sppm_channel, sppm_channel, sizes)

        self.uafms = nn.ModuleList()
        self.uafms.append(UnifiedAttentionFusionModule(in_channels[0], out_channels[1], out_channels[0]))
        self.uafms.append(UnifiedAttentionFusionModule(in_channels[1], out_channels[2], out_channels[1]))
        self.uafms.append(UnifiedAttentionFusionModule(in_channels[2], out_channels[2], out_channels[2]))

        self.classifiers = nn.ModuleList()  # [..., head_16, head32]
        mid_ch = out_channels[1]
        for in_ch in out_channels:
            self.classifiers.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, num_classes, 1, bias=False)
                )
            )

        self._init_weight()

    def forward(self, feats):
        outs = []
        high_feat = self.sppm(feats[-1])
        for low_feat, uafm, classifier in zip(reversed(feats), reversed(self.uafms), reversed(self.classifiers)):
            high_feat = uafm(low_feat, high_feat)
            outs.append(classifier(high_feat))
        return outs

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    model = PPLiteSegHead(sizes=[1, 2, 4], in_channels=[256, 512, 1024],
                          sppm_channel=128, out_channels=[32, 64, 128], num_classes=19)
    print(model)

    in_channels = [256, 512, 1024]
    scales = [64, 32, 16]
    inputs = [torch.rand(2, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')

    outputs = model(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')