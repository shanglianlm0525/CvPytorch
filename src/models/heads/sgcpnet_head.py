# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/15 9:44
# @Author : liumin
# @File : sgcpnet_head.py

import torch
from torch import nn
from src.models.modules.activations import MemoryEfficientSwish
from src.models.modules.convs import DepthwiseConvModule
from src.utils.torch_utils import set_bn_momentum


class SGCPNetHead(nn.Module):
    def __init__(self, num_classes, in_channels, shrink_channels, mid_channels, first_time=True):
        super(SGCPNetHead, self).__init__()
        self.num_classes = num_classes
        self.mid_channels = mid_channels
        self.first_time = first_time
        self.epsilon = 1e-4
        self.relu = nn.ReLU(inplace=False)
        self.swish = MemoryEfficientSwish()

        self.conv3 = nn.Conv2d(in_channels[0], shrink_channels[0], 1, 1, 0)
        self.conv4 = nn.Conv2d(in_channels[1], shrink_channels[1], 1, 1, 0)
        self.conv5 = nn.Conv2d(in_channels[2], shrink_channels[2], 1, 1, 0)

        if self.first_time:
            self.p5_to_p6 = nn.Sequential(
                nn.Conv2d(shrink_channels[2], self.mid_channels, 1),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
                nn.MaxPool2d(3, 2, 1)
            )
            self.p6_to_p7 = nn.Sequential(
                nn.MaxPool2d(3, 2, 1)
            )

            self.p3_down_channel = nn.Sequential(
                nn.Conv2d(shrink_channels[0], self.mid_channels, 1, 1, 0),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
            )

            self.p4_down_channel = nn.Sequential(
                nn.Conv2d(shrink_channels[1], self.mid_channels, 1, 1, 0),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_down_channel = nn.Sequential(
                nn.Conv2d(shrink_channels[2], self.mid_channels, 1, 1, 0),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
            )


            self.p4_down_channel_2 = nn.Sequential(
                nn.Conv2d(shrink_channels[1], self.mid_channels, 1, 1, 0),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
            )

            self.p5_down_channel_2 = nn.Sequential(
                nn.Conv2d(shrink_channels[2], self.mid_channels, 1, 1, 0),
                nn.BatchNorm2d(self.mid_channels, momentum=0.01, eps=1e-3),
            )

        # top-down path
        self.conv6_up = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv5_up = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv4_up = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv3_up = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)

        self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # Botton-up path
        self.conv4_down = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv5_down = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv6_down = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv7_down = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)

        self.p4_downsample = nn.MaxPool2d(3, 2, 1)
        self.p5_downsample = nn.MaxPool2d(3, 2, 1)
        self.p6_downsample = nn.MaxPool2d(3, 2, 1)
        self.p7_downsample = nn.MaxPool2d(3, 2, 1)

        self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        # top-down path
        self.conv6_up_2 = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv5_up_2 = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv4_up_2 = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)
        self.conv3_up_2 = DepthwiseConvModule(self.mid_channels, self.mid_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation=None)

        self.p6_w1_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w1_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

        self.classifier = nn.Conv2d(self.mid_channels, self.num_classes, 1, 1, 0)

        self._init_weight()

        set_bn_momentum(self, momentum=0.01, eps=1e-3)

    def forward(self, x):
        if self.first_time:
            p3, p4, p5 = x

            p3 = self.conv3(p3)
            p4 = self.conv4(p4)
            p5 = self.conv5(p5)

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = x

        # Top-down path
        p6_w1 = self.relu(self.p6_w1)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_SW_sum_1 = self.swish(weight[0] * p6_in + weight[1] *nn.Upsample(size=p6_in.size()[2:], mode='nearest')(p7_in))
        p6_up = self.conv6_up(p6_SW_sum_1)

        p5_w1 = self.relu(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_SW_sum_1 = self.swish(weight[0] * p5_in + weight[1] *nn.Upsample(size=p5_in.size()[2:], mode='nearest')(p6_up))
        p5_up = self.conv5_up(p5_SW_sum_1)

        p4_w1 = self.relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_SW_sum_1 = self.swish(weight[0] * p4_in + weight[1] *nn.Upsample(size=p4_in.size()[2:], mode='nearest')(p5_up))
        p4_up = self.conv4_up(p4_SW_sum_1)

        p3_w1 = self.relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_SW_sum_1 = self.swish(weight[0] * p3_in + weight[1] *nn.Upsample(size=p3_in.size()[2:], mode='nearest')(p4_up))
        p3_out = self.conv3_up(p3_SW_sum_1)

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Bottom-up path
        p4_w2 = self.relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_SW_sum_2 = self.swish(
            weight[0] * p4_in + weight[1] * nn.Upsample(size=p4_in.size()[2:])(p4_up) + weight[2] * nn.Upsample(
                size=p4_in.size()[2:])(self.p4_downsample(p3_out)))
        p4_out = self.conv4_down(p4_SW_sum_2)

        p5_w2 = self.relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_SW_sum_2 = self.swish(
            weight[0] * p5_in + weight[1] * nn.Upsample(size=p5_in.size()[2:])(p5_up) + weight[2] * nn.Upsample(
                size=p5_in.size()[2:])(self.p5_downsample(p4_out)))
        p5_out = self.conv5_down(p5_SW_sum_2)

        p6_w2 = self.relu(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
        p6_SW_sum_2 = self.swish(
            weight[0] * p6_in + weight[1] * nn.Upsample(size=p6_in.size()[2:])(p6_up) + weight[2] * nn.Upsample(
                size=p6_in.size()[2:])(self.p6_downsample(p5_out)))
        p6_out = self.conv6_down(p6_SW_sum_2)

        p7_w2 = self.relu(self.p7_w2)
        weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
        p7_SW_sum_2 = self.swish(
            weight[0] * p7_in + weight[1] * nn.Upsample(size=p7_in.size()[2:])(self.p7_downsample(p6_out)))
        p7_out = self.conv7_down(p7_SW_sum_2)

        # Top-down path
        p6_w1 = self.relu(self.p6_w1_2)
        weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
        p6_SW_sum_1 = self.swish(weight[0] * p6_out + weight[1] *nn.Upsample(size=p6_out.size()[2:], mode='nearest')(p7_out))
        p6_up = self.conv6_up_2(p6_SW_sum_1)

        p5_w1 = self.relu(self.p5_w1_2)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
        p5_SW_sum_1 = self.swish(weight[0] * p5_out + weight[1] *nn.Upsample(size=p5_out.size()[2:], mode='nearest')(p6_up))
        p5_up = self.conv5_up_2(p5_SW_sum_1)

        p4_w1 = self.relu(self.p4_w1_2)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_SW_sum_1 = self.swish(weight[0] * p4_out + weight[1] *nn.Upsample(size=p4_out.size()[2:], mode='nearest')(p5_up))
        p4_up = self.conv4_up_2(p4_SW_sum_1)

        p3_w1 = self.relu(self.p3_w1_2)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_SW_sum_1 = self.swish(weight[0] * p3_out + weight[1] *nn.Upsample(size=p3_out.size()[2:], mode='nearest')(p4_up))
        p3_out = self.conv3_up_2(p3_SW_sum_1)

        return self.classifier(p3_out)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)