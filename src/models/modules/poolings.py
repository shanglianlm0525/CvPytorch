# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/26 17:55
# @Author : liumin
# @File : poolings.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__  = ['SoftPool1D', 'SoftPool2D', 'SoftPool3D', 'GlobalAvgPool2d', 'ConcatDownsample2d', 'SPP', 'BlurPool2D', 'BlurPool1D']


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


class GlobalAvgPool2d(nn.Module):
    """ Fast implementation of global average pooling from
        TResNet: High Performance GPU-Dedicated Architecture
        https://arxiv.org/pdf/2003.13630.pdf
    Args:
        flatten (bool, optional): whether spatial dimensions should be squeezed
    """
    def __init__(self, flatten: bool = False) -> None:
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class ConcatDownsample2d(nn.Module):
    '''
        Implements a loss-less downsampling operation by stacking adjacent information on the channel dimension.
        YOLO9000: Better, Faster, Stronger
        https://pjreddie.com/media/files/papers/YOLO9000.pdf

        Args:
            x (torch.Tensor[N, C, H, W]): input tensor
            scale_factor (int): spatial scaling factor

        Returns:
            torch.Tensor[N, scale_factor ** 2 * C, H / scale_factor, W / scale_factor]: downsampled tensor
    '''
    def __init__(self, scale_factor):
        super(ConcatDownsample2d, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        b, c, h, w = x.shape

        if (h % self.scale_factor != 0) or (w % self.scale_factor != 0):
            raise AssertionError("Spatial size of input tensor must be multiples of `scale_factor`")

        # N * C * H * W --> N * C * (H/scale_factor) * scale_factor * (W/scale_factor) * scale_factor
        x = x.view(b, c, h // self.scale_factor, self.scale_factor, w // self.scale_factor, self.scale_factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(b, int(c * self.scale_factor ** 2), h // self.scale_factor, w // self.scale_factor)
        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
        https://arxiv.org/pdf/1406.4729.pdf
    """
    def __init__(self, kernel_sizes):
        super().__init__()
        self.pools = [nn.MaxPool2d(k_size, stride=1, padding=k_size // 2) for k_size in kernel_sizes]

    def forward(self, x):
        feats = [x] + [pool(x) for pool in self.pools]
        return torch.cat(feats, dim=1)


def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class BlurPool2D(nn.Module):
    """
        Making Convolutional Networks Shift-Invariant Again
        https://arxiv.org/abs/1904.11486
    """
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool2D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

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
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return


class BlurPool1D(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(BlurPool1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
