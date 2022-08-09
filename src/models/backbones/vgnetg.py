# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/8/7 19:02
# @Author : liumin
# @File : vgnetg.py

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
    Efficient CNN Architecture Design Guided by Visualization
    https://arxiv.org/pdf/2207.10318.pdf
"""

class ChannelSplit(nn.Module):
    def __init__(self, sections, dim=1):
        super().__init__()
        self.sections = sections
        self.dim = dim

    def forward(self, x):
        return torch.split(x, self.sections, dim=self.dim)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, dim=self.dim)


class ChannelChunk(nn.Module):
    def __init__(self, groups, dim=1):
        super().__init__()
        self.groups = groups
        self.dim = dim

    def forward(self, x):
        return torch.chunk(x, self.groups, dim=self.dim)


class SharedDepthwiseConv2d(nn.Module):
    def __init__(self, channels, kernel_size = 3, stride = 1, padding = None, dilation = 1, t = 2, bias = False):
        super().__init__()
        self.channels = channels // t
        self.t = t

        if padding is None:
            padding = ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.mux = nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size, stride, padding, dilation, groups=self.channels, bias=bias),
                nn.BatchNorm2d(self.channels),
                nn.SiLU(inplace=True)
            )

    def forward(self, x):
        x = torch.chunk(x, self.t, dim=1)
        x = [self.mux(xi) for xi in x]
        return torch.cat(x, dim=1)


class SEModule(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(SEModule, self).__init__()
        mid_channel = channel // ratio
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(in_features=channel, out_features=mid_channel),
                nn.SiLU(inplace=True),
                nn.Linear(in_features=mid_channel, out_features=channel),
                nn.Sigmoid()
            )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


def get_gaussian_kernel1d(kernel_size, sigma: torch.Tensor):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size).to(sigma.device)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    return pdf / pdf.sum()


def get_gaussian_kernel2d(kernel_size, sigma: torch.Tensor):
    kernel1d = get_gaussian_kernel1d(kernel_size, sigma)
    return torch.mm(kernel1d[:, None], kernel1d[None, :])

class GaussianBlur(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None,
        dilation: int = 1,
        sigma: float = 1.0,
        learnable: bool = True
    ):
        super().__init__()

        padding = padding or ((kernel_size - 1) * (dilation - 1) + kernel_size) // 2

        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.dilation = (dilation, dilation)
        self.padding_mode = 'zeros'
        self.learnable = learnable

        self.sigma = nn.Parameter(torch.tensor(sigma), learnable)

    def forward(self, x):
        # print(f'--> {self.channels:>3d}, {self.sigma.item():>6.3f}')
        return F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.channels)

    @property
    def weight(self):
        kernel = get_gaussian_kernel2d(self.kernel_size[0], self.sigma)
        return kernel.repeat(self.channels, 1, 1, 1)

    @property
    def out_channels(self):
        return self.channels


class DownsamplingBlock(nn.Module):
    def __init__(self, inp, oup, stride=2, method='blur', se_ratio=0):
        super(DownsamplingBlock, self).__init__()
        assert method in ['blur', 'dwconv', 'maxpool'], f'{method}'

        if method == 'dwconv' or stride == 1:
            self.downsample = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp)
        elif method == 'maxpool':
            self.downsample = nn.MaxPool2d(kernel_size=3, stride=stride)
        elif method == 'blur':
            self.downsample = GaussianBlur(inp, stride=stride, sigma=1.1, learnable=False)
        else:
            ValueError(f'Unknown downsampling method: {method}.')

        split_chs = 0 if inp > oup else min(oup // 2, inp)

        self.split = ChannelSplit([inp - split_chs, split_chs])
        self.conv1x1 = nn.Sequential(
                            nn.Conv2d(inp, oup - split_chs, 1, 1, 0),
                            nn.BatchNorm2d(oup - split_chs),
                            nn.SiLU(inplace=True)
                        )

        if se_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(inp, oup - split_chs, 1, 1, 0),
                nn.BatchNorm2d(oup - split_chs),
                nn.SiLU(inplace=True),
                SEModule(oup - split_chs, se_ratio)
            )

        self.halve = nn.Identity()
        if oup > 2 * inp or inp > oup:
            self.halve = nn.Sequential(
                Concat(dim=1),
                ChannelChunk(2)
            )

    def forward(self, x):
        x = self.downsample(x)
        _, x2 = self.split(x)
        return self.halve([x2, self.conv1x1(x)])


class HalfIdentityBlock(nn.Module):
    def __init__(self,  inp, se_ratio= 0):
        super(HalfIdentityBlock, self).__init__()
        self.half3x3 = nn.Conv2d(inp // 2, inp // 2, 3, 1, 1, groups=(inp // 2))
        self.combine = Concat(dim=1)
        self.conv1x1 = nn.Sequential(
                            nn.Conv2d(inp, inp // 2, 1, 1, 0),
                            nn.BatchNorm2d(inp // 2),
                            nn.SiLU(inplace=True)
                        )

        if se_ratio > 0.0:
            self.conv1x1 = nn.Sequential(
                nn.Conv2d(inp, inp // 2, 1, 1, 0),
                nn.BatchNorm2d(inp // 2),
                nn.SiLU(inplace=True),
                SEModule(inp // 2, se_ratio)
            )

    def forward(self, x):
        out = self.combine([x[0], self.half3x3(x[1])])
        return [x[1], self.conv1x1(out)]


class VGNetG(nn.Module):
    def __init__(self, subtype='vgnetg_x1.0', out_stages=[2, 3, 4], output_stride=32, classifier=False, num_classes=1000, backbone_path=None, pretrained = False):
        super(VGNetG, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'vgnetg_x1.0':
            channels= [28, 56, 112, 224, 368]
            blocks = [4, 7, 13, 2]
            se_ratio=4
        elif self.subtype == 'vgnetg_x1.5':
            channels= [32, 64, 128, 256, 512]
            blocks = [3, 7, 14, 2]
            se_ratio=4
        elif self.subtype == 'vgnetg_x2.0':
            channels= [32, 72, 168, 376, 512]
            blocks = [3, 6, 13, 2]
            se_ratio=4
        elif self.subtype == 'vgnetg_x2.5':
            channels= [32, 80, 192, 400, 544]
            blocks = [3, 6, 16, 2]
            se_ratio=4
        elif self.subtype == 'vgnetg_x5.0':
            channels= [32, 88, 216, 456, 856]
            blocks = [4, 7, 15, 5]
            se_ratio=4
        else:
            raise NotImplementedError


        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1)
        )

        self.layer1 = self.make_layer(channels[0], channels[1], 'blur', blocks[0], se_ratio)
        self.layer2 = self.make_layer(channels[1], channels[2], 'blur', blocks[1], se_ratio)
        self.layer3 = self.make_layer(channels[2], channels[3], 'blur', blocks[2], se_ratio)
        self.layer4 = self.make_layer(channels[3], channels[4], 'blur', blocks[3], se_ratio)

        self.layer4.append(nn.Sequential(
            SharedDepthwiseConv2d(channels[-1], t=8),
            nn.Conv2d(channels[-1], channels[-1], 1, 1, 0),
            nn.BatchNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
        ))

        if self.classifier:
            self.avg = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(channels[-1], num_classes)

    def make_layer(self, in_places, places, method, block, se_ratio):
        layers = [DownsamplingBlock(in_places, places, stride=2, method=method, se_ratio=se_ratio)]
        for _ in range(block - 1):
            layers.append(HalfIdentityBlock(places, se_ratio))

        layers.append(Concat(dim=1))
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            res_layer = getattr(self, 'layer{}'.format(i))
            x = res_layer(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)

        if self.classifier:
            x = self.avg(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return output if len(self.out_stages) > 1 else output[0]


if __name__ == "__main__":
    model = VGNetG('vgnetg_x1.0', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)