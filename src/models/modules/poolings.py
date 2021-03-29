# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/26 17:55
# @Author : liumin
# @File : poolings.py

import torch
import torch.nn as nn

__all__  = ['SoftPool1D', 'SoftPool2D', 'SoftPool3D']


class SoftPool1D(torch.nn.Module):
    def __init__(self,kernel_size=2,stride=2):
        super(SoftPool1D, self).__init__()
        self.avgpool = torch.nn.AvgPool1d(kernel_size,stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool


class SoftPool2D(nn.Module):
    """
        Refining activation downsampling with SoftPool
        PDF: https://arxiv.org/pdf/2101.00440v3.pdf
    """
    def __init__(self, kernel_size=2, stride=2):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class SoftPool3D(torch.nn.Module):
    def __init__(self,kernel_size,stride=2):
        super(SoftPool3D, self).__init__()
        self.avgpool = nn.AvgPool3d(kernel_size,stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp*x)
        return x/x_exp_pool