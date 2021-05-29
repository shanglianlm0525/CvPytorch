# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/26 17:55
# @Author : liumin
# @File : quant_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizers(nn.Module):
    def __init__(self, bw, quant_mode='mse', act_q=True, quantize=False):
        super(Quantizers, self).__init__()
        self.is_quantize = quantize
        self.act_q = act_q
        self.init = False
        self.is_symmetric = False
        self.quant_mode = quant_mode
        self.calibration = False
        self.n = bw
        self.offset = None
        self.min = torch.Tensor([float('inf')])[0].cuda()
        self.max = torch.Tensor([float('-inf')])[0].cuda()
        self.scale = None
        self.min_mse = float('inf')

    def set_quantize(self, flag):
        self.is_quantize = flag

    def estimate_range(self, flag):
        self.calibration = flag

    def init_params(self, x_f):
        '''
        https://heartbeat.fritz.ai/quantization-arithmetic-421e66afd842

        There exist two modes
        1) Symmetric:
            Symmetric quantization uses absolute max value as its min/max meaning symmetric with respect to zero
        2) Asymmetric
            Asymmetric Quantization uses actual min/max, meaning it is asymmetric with respect to zero

        Scale factor uses full range [-2**n / 2, 2**n - 1]
        '''

        if self.is_symmetric:
            x_min, x_max = -torch.max(torch.abs(x_f)), torch.max(torch.abs(x_f))
        else:
            x_min, x_max = torch.min(x_f), torch.max(x_f)

        self.min = torch.min(x_min, self.min)
        self.max = torch.max(x_max, self.max)
        max_range = self.max - self.min

        if self.quant_mode == 'mse':
            if not self.init or self.act_q:
                self.init = True
                for i in range(80):
                    scale = (max_range - (0.01 * i)) / float(2 ** self.n - 1)
                    offset = torch.round(-self.min / scale)
                    x_fp_q = self.quant_dequant(x_f, scale, offset)
                    curr_mse = torch.pow(x_fp_q - x_f, 2).mean().cpu().numpy()
                    if self.min_mse > curr_mse:
                        self.min_mse = curr_mse
                        self.scale = scale
                        self.offset = offset

        elif self.quant_mode == 'minmax':
            self.scale = max_range / float(2 ** self.n - 1)
            if not self.is_symmetric:
                self.offset = torch.round(-x_min / self.scale)
        self.init = True

    def quant_dequant(self, x_f, scale, offset):
        '''
        Quantizing
        Formula is derived from below:
        https://medium.com/ai-innovation/quantization-on-pytorch-59dea10851e1
        '''
        x_int = torch.round(x_f / scale)
        if not self.is_symmetric:
            x_int += offset

        if self.is_symmetric:
            l_bound, u_bound = -2 ** (self.n - 1), 2 ** (self.n - 1) - 1
        else:
            l_bound, u_bound = 0, 2 ** (self.n) - 1
        x_q = torch.clamp(x_int, min=l_bound, max=u_bound)

        '''
        De-quantizing
        '''
        if not self.is_symmetric:
            x_q -= offset
        x_float_q = x_q * scale
        return x_float_q

    def forward(self, x_f):
        if (self.calibration and self.act_q) or not self.init:
            self.init_params(x_f)
        return self.quant_dequant(x_f, self.scale, self.offset) if self.is_quantize else x_f


class QConv2d(nn.Module):
    def __init__(self, conv, norm=None, act=None, w_scheme='mse', a_scheme='mse', w_bit = 8, b_bit=8, a_bit=8):
        super(QConv2d, self).__init__()
        self.conv = conv
        self.weight_quantizer = Quantizers(w_bit, w_scheme, act_q=False)
        self.kwarg = {'stride': self.conv.stride, 'padding': self.conv.padding,
                      'dilation': self.conv.dilation, 'groups': self.conv.groups}
        self.norm = norm
        '''
        if self.norm is not None:
            bn_state_dict = self.norm.state_dict()
            self.conv.register_buffer('eps', torch.tensor(self.norm.eps))
            self.conv.register_buffer('gamma', bn_state_dict['weight'].detach())
            self.conv.register_buffer('beta', bn_state_dict['bias'].detach())
            self.conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
            self.conv.register_buffer('var', bn_state_dict['running_var'].detach())
        '''
        self.act = act
        self.act_quantizer = Quantizers(a_bit, a_scheme)
        self.pre_act = False

    def fuse(self):
        pass

    def forward(self, x):
        w, b = self.get_params()
        out = F.conv2d(input=x, weight=w, bias=b, **self.kwarg)
        if self.act and not self.pre_act:
            out = self.act(out)
        out = self.act_quantizer(out)
        return out


class QLinear(nn.Module):
    def __init__(self, linear, norm=None, act=None, w_scheme='mse', a_scheme='mse', w_bit = 8, b_bit=8, a_bit=8):
        super(QLinear, self).__init__()
        self.fc = linear
        self.weight_quantizer = Quantizers(w_bit, w_scheme, act_q = False)
        self.norm = norm
        self.act = act
        self.act_quantizer = Quantizers(a_bit, a_scheme)

    def get_params(self):
        w = self.fc.weight.detach()
        if self.fc.bias != None:
            b = self.fc.bias.detach()
        else:
            b = None
        w = self.weight_quantizer(w)
        return w, b

    def forward(self, x):
        w, b = self.get_params()
        out = F.linear(x, w, b)
        if self.act:
            out = self.act(out)
        out = self.act_quantizer(out)
        return out


class QIdentity(nn.Module):
    def __init__(self):
        super(QIdentity, self).__init__()

    def forward(self, x):
        return x


class QConcat(nn.Module):
    def __init__(self, dim):
        super(QConcat, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x,y),self.dim)


class QAdd(nn.Module):
    def __init__(self):
        super(QAdd, self).__init__()

    def forward(self, x, y):
        return x + y