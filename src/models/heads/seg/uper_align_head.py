# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/31 18:24
# @Author : liumin
# @File : uper_align_head.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.bricks import ConvModule
from src.models.heads.seg.base_seg_head import BaseSegHead
from src.models.heads.seg.psp_head import PPM


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature = self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class AlignedModulev2(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModulev2, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 4, kernel_size=kernel_size, padding=1, bias=False)
        self.flow_gate = nn.Sequential(
            nn.Conv2d(outplane * 2, 1, kernel_size=kernel_size, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        l_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)

        flow = self.flow_make(torch.cat([h_feature, l_feature], 1))
        flow_up, flow_down = flow[:, :2, :, :] , flow[:, 2:, :, :]

        h_feature_warp = self.flow_warp(h_feature_orign, flow_up, size=size)
        l_feature_warp = self.flow_warp(low_feature, flow_down, size=size)

        flow_gates = self.flow_gate(torch.cat([h_feature, l_feature], 1))

        fuse_feature = h_feature_warp * flow_gates + l_feature_warp * (1 - flow_gates)

        return fuse_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid, align_corners=True)
        return output


class UperNetAlignHead(BaseSegHead):
    """Unified Perceptual Parsing for Scene Understanding.
    Args:
        bins (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, bins=(1, 2, 3, 6), **kwargs):
        super(UperNetAlignHead, self).__init__(**kwargs)

        # PSP Module
        self.ppm = PPM(bins, self.in_channels[-1], self.channels,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.ppm_bottleneck = ConvModule(
            self.in_channels[-1] + len(bins) * self.channels,
            self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_aligns = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)
            fpn_conv = ConvModule(self.channels, self.channels, 3, padding=1,
                conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False)

            fpn_align = AlignedModule(inplane=self.channels, outplane=self.channels // 2)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_aligns.append(fpn_align)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels, self.channels, 3, padding=1,
            conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)


    def forward(self, x):
        """Forward function."""

        # build laterals
        laterals = [
            lateral_conv(x[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Forward function of PSP module.
        psp_outs = [x[-1]]
        psp_outs.extend(self.ppm(x[-1]))
        psp_outs = torch.cat(psp_outs, dim=1)
        psp_outs = self.ppm_bottleneck(psp_outs)
        laterals.append(psp_outs)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            '''
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=False)
            '''
            laterals[i - 1] = laterals[i - 1] + self.fpn_aligns[i-1]([laterals[i - 1], laterals[i]])
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode='bilinear', align_corners=False)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_outs = self.fpn_bottleneck(fpn_outs)
        return self.classify(fpn_outs)
