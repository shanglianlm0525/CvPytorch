# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/2/17 19:05
# @Author : liumin
# @File : regnet.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import ConvModule

"""
    Rethink Dilated Convolution for Real-time Semantic Segmentation
    https://arxiv.org/pdf/2111.09957.pdf
"""


class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2,2,ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)
        res=[]
        for i in range(self.num_splits):
            res.append(self.convs[i](x[i]))
        return torch.cat(res,dim=1)


class SEModule(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SEModule, self).__init__()
        mid_channel = out_channel // 4
        self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channel, mid_channel, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channel, in_channel, 1, bias=True),
                nn.Sigmoid()
            )
    def forward(self, x):
        return x * self.se(x)


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,
                 group_width, stride, attention="se"):
        super().__init__()
        avg_downsample = True
        groups = out_channels // group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                stride=stride, groups=groups, padding=dilation, dilation=dilation, bias=False)
        else:
            self.conv2 = DilatedConv(out_channels, dilations, group_width=group_width, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU(inplace=True)
        if attention == "se":
            self.se = SEModule(out_channels, in_channels)
        else:
            self.se = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Shortcut(in_channels, out_channels, stride, avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut = self.shortcut(x) if self.shortcut else x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x + shortcut)
        return x


class RegSegBackbone(nn.Module):
    def __init__(self, subtype='', out_channels = [32, 48, 128, 320], out_stages=[1, 2, 3], output_stride=16,
                 classifier=False, backbone_path=None, pretrained=False):
        super(RegSegBackbone, self).__init__()
        self.subtype = subtype
        self.out_channels = out_channels
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        attention = "se"
        ds = [[1], [1, 2]] + 4 * [[1, 4]] + 7 * [[1, 14]]
        gw = 16

        self.stem = ConvModule(3, self.out_channels[0], 3, 2, 1, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'))
        self.stage1 = DBlock(self.out_channels[0], self.out_channels[1], [1], gw, 2, attention)  # 4
        self.stage2 = nn.Sequential(
            DBlock(self.out_channels[1], self.out_channels[2], [1], gw, 2, attention),
            DBlock(self.out_channels[2], self.out_channels[2], [1], gw, 1, attention),
            DBlock(self.out_channels[2], self.out_channels[2], [1], gw, 1, attention)
        )  # 8
        self.stage3 = nn.Sequential(
            DBlock(self.out_channels[2], 256, [1], gw, 2, attention),
            *self.generate_stage(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention)),
            DBlock(256, self.out_channels[3], ds[-1], gw, 1, attention)
        )  # 16

    def generate_stage(self, ds, block_fun):
        blocks = []
        for d in ds:
            blocks.append(block_fun(d))
        return blocks

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 4):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    model = RegSegBackbone('')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)
