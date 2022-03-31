# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/3/28 16:24
# @Author : liumin
# @File : lfd_resnet.py

import torch
import torch.nn as nn


class FastBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FastBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self._conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1)
        self._norm1 = nn.BatchNorm2d(self.out_channels)

        self._conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1, stride=1, padding=0)
        self._norm2 = nn.BatchNorm2d(self.out_channels)
        self._activation = nn.ReLU(inplace=True)

        self._conv3 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self._norm3 = nn.BatchNorm2d(self.out_channels)

        if self.stride > 1 :
            self._downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        out = self._norm2(out)
        out = self._activation(out)

        out = self._conv3(out)
        out = self._norm3(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class FasterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FasterBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self._conv1 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1)
        self._norm1 = nn.BatchNorm2d(self.out_channels)

        self._activation = nn.ReLU(inplace=True)

        self._conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self._norm2 = nn.BatchNorm2d(self.out_channels)

        if self.stride > 1 :
            self._downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        out = self._norm2(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class FastestBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FastestBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        mid_channels = out_channels // 2
        self._conv1 = nn.Conv2d(self.in_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1)
        self._norm1 = nn.BatchNorm2d(self.out_channels)

        self._activation = nn.ReLU(inplace=True)

        self._conv2 = nn.Conv2d(mid_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self._norm2 = nn.BatchNorm2d(self.out_channels)

        if self.stride > 1 :
            self._downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(self.out_channels)
            )

    def forward(self, x):
        identity = x

        out = self._conv1(x)
        out = self._norm1(out)
        out = self._activation(out)

        out = self._conv2(out)
        out = self._norm2(out)

        if self._downsample is not None:
            identity = self._downsample(x)
        out += identity
        out = self._activation(out)

        return out


class LFDResNet(nn.Module):
    def __init__(self, subtype='lfd_s', out_stages=[2, 3, 4], output_stride=32, backbone_path=None,
                 pretrained=False):
        super(LFDResNet, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'lfd_xs':
            block_mode = 'faster'
            self.block_num = [4, 2, 2, 3]
            self.out_channels = [32, 64, 64, 64, 64]
        elif self.subtype == 'lfd_s':
            block_mode = 'faster'
            self.block_num = [4, 2, 2, 3]
            self.out_channels = [64, 64, 64, 64, 128]
        elif self.subtype == 'lfd_m':
            block_mode = 'faster'
            self.block_num = [3, 2, 1, 1, 1]
            self.out_channels = [64, 64, 64, 64, 128, 128]
        elif self.subtype == 'lfd_l':
            block_mode = 'faster'
            self.block_num = [4, 2, 2, 1, 1]
            self.out_channels = [64, 64, 64, 64, 128, 128]
        else:
            raise NotImplementedError

        self.stem = nn.Sequential(
            nn.Conv2d(3, self.out_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels[0], self.out_channels[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU(inplace=True),
        )

        self._make_layer()

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        if self.pretrained:
            self.load_pretrained_weights()
        else:
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

    def _make_layer(self):
        for i, num_blocks in enumerate(self._body_architecture):
            num_stage_channels = self._body_channels[i]
            stage_list = nn.ModuleList()
            in_channels = self._stem_channels if i == 0 else self._body_channels[i - 1]
            for j in range(num_blocks):
                if j == 0:
                    stage_list.append(self._block(in_channels, num_stage_channels, stride=2))
                else:
                    stage_list.append(self._block(num_stage_channels, num_stage_channels, stride=1))

            setattr(self, 'layer%d' % i, stage_list)

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages:
                output.append(x)

        return output if len(self.out_stages) > 1 else output[0]


if __name__=="__main__":
    model = LFDResNet()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)