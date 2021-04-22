# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/26 17:55
# @Author : liumin
# @File : poolings.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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


"""
    Making Convolutional Networks Shift-Invariant Again
    https://arxiv.org/abs/1904.11486
"""
class BlurPool(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        # 定义一系列的高斯核
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt) # 归一化操作，保证特征经过blur后信息总量不变
        # 非grad操作的参数利用buffer存储
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            # 利用固定参数的conv2d+stride实现blurpool
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])