# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/14 21:18
# @Author : liumin
# @File : custom_cspnet.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0

from src.models.modules.convs import ConvModule

"""
    CSPNET: A NEW BACKBONE THAT CAN ENHANCE LEARNING CAPABILITY OF CNN
    https://arxiv.org/pdf/1911.11929.pdf
"""

class TinyResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, norm_cfg, activation, res_type='concat'):
        super(TinyResBlock, self).__init__()
        assert in_channels % 2 == 0
        assert res_type in ['concat', 'add']
        self.res_type = res_type
        self.in_conv = ConvModule(in_channels, in_channels//2, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        self.mid_conv = ConvModule(in_channels//2, in_channels//2, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        if res_type == 'add':
            self.out_conv = ConvModule(in_channels//2, in_channels, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.mid_conv(x)
        if self.res_type == 'add':
            return self.out_conv(x+x1)
        else:
            return torch.cat((x1, x), dim=1)


class CspBlock(nn.Module):
    def __init__(self, in_channels, num_res, kernel_size=3, stride=0, norm_cfg=dict(type='BN', requires_grad=True), activation='LeakyReLU'):
        super(CspBlock, self).__init__()
        assert in_channels % 2 == 0
        self.in_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)
        res_blocks = []
        for i in range(num_res):
            res_block = TinyResBlock(in_channels, kernel_size, norm_cfg, activation)
            res_blocks.append(res_block)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.res_out_conv = ConvModule(in_channels, in_channels, kernel_size, padding=(kernel_size-1)//2, norm_cfg=norm_cfg, activation=activation)

    def forward(self, x):
        x = self.in_conv(x)
        x1 = self.res_blocks(x)
        x1 = self.res_out_conv(x1)
        out = torch.cat((x1, x), dim=1)
        return out


class CustomCspNet(nn.Module):

    def __init__(self, subtype='cspnet', out_stages=[2,3,4], output_stride=32, backbone_path=None, pretrained=True):
        super(CustomCspNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'cspnet':

            net_cfg = [['Conv', 3, 32, 3, 2],  # 1/2
                       ['MaxPool', 3, 2],  # 1/4
                       ['CspBlock', 32, 1, 3, 1],  # 1/4
                       ['CspBlock', 64, 2, 3, 2],  # 1/8
                       ['CspBlock', 128, 2, 3, 2],  # 1/16
                       ['CspBlock', 256, 3, 3, 2]]  # 1/32

            self.stages = nn.ModuleList()
            for stage_cfg in net_cfg:
                if stage_cfg[0] == 'Conv':
                    in_channels, out_channels, kernel_size, stride = stage_cfg[1:]
                    stage = ConvModule(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2,
                                       norm_cfg=dict(type='BN', requires_grad=True), activation='LeakyReLU')
                elif stage_cfg[0] == 'CspBlock':
                    in_channels, num_res, kernel_size, stride = stage_cfg[1:]
                    stage = CspBlock(in_channels, num_res, kernel_size, stride, dict(type='BN', requires_grad=True), 'LeakyReLU')
                elif stage_cfg[0] == 'MaxPool':
                    kernel_size, stride = stage_cfg[1:]
                    stage = nn.MaxPool2d(kernel_size, stride, padding=(kernel_size - 1) // 2)
                else:
                    raise ModuleNotFoundError
                self.stages.append(stage)

            self.out_channels = [32, 32, 64, 128, 256]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()

    def forward(self, x):
        output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output) if len(self.out_stages) > 1 else output[0]


    def init_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)



if __name__=="__main__":
    model =CustomCspNet('cspnet')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)