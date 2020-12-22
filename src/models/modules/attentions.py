# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/21 18:59
# @Author : liumin
# @File : attentions.py

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
        Squeeze-and-Excitation Networks
        https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        mid_channel = channel // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class SKBlock(nn.Module):
    def __init__(self):
        super(SKBlock, self).__init__()

    def forward(self, x):
        pass


class sSEBlock(nn.Module):
    def __init__(self, channel):
        super(sSEBlock, self).__init__()
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class cSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(cSEBlock, self).__init__()
        mid_channel = channel // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)


class scSEBlock(nn.Module):
    """
        Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
        https://arxiv.org/pdf/1803.02579v2.pdf
    """
    def __init__(self, channel, reduction=16):
        super(scSEBlock, self).__init__()
        self.cSE = cSEBlock(channel, reduction)
        self.sSE = sSEBlock(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class NonLocalBlock(nn.Module):
    """
        Non-local Neural Networks
        https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                  bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)


    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class ContextBlock(nn.Module):
    def __init__(self):
        super(ContextBlock, self).__init__()

    def forward(self, x):
        pass


class CrissCrossAttention(nn.Module):
    def __init__(self):
        super(CrissCrossAttention, self).__init__()

    def forward(self, x):
        pass


class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()

    def forward(self, x):
        pass


class BAM(nn.Module):
    def __init__(self):
        super(BAM, self).__init__()

    def forward(self, x):
        pass


class SplitAttention(nn.Module):
    def __init__(self):
        super(SplitAttention, self).__init__()

    def forward(self, x):
        pass


if __name__=='__main__':
    # model = SEBlock(16)
    # model = scSEBlock(16, 4)
    model = NonLocalBlock(16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)