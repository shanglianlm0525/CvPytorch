# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/23 13:27
# @Author : liumin
# @File : bn_fusion.py

import torch
import torch.nn as nn

def fuse_bn_sequential(block):
    """
        y = gamma*(conv(x,w)-mu)/sqrt(var+epsilon)+beta
    """
    stack = []
    flag = 0
    temp_conv = None
    for m in block.children():
        if isinstance(m, nn.Conv2d):
            temp_conv = m
        elif isinstance(m, nn.BatchNorm2d) and isinstance(temp_conv,nn.Conv2d):
            flag = 1
            bn_st_dict = m.state_dict()
            conv_st_dict = temp_conv.state_dict()
            # BatchNorm params
            eps   = m.eps
            mu    = bn_st_dict['running_mean']
            var   = bn_st_dict['running_var']
            gamma = bn_st_dict['weight']
            beta  = bn_st_dict['bias']
            # Conv params
            weight = conv_st_dict['weight']
            # fusion the params
            denom = torch.sqrt(var + eps)
            A = gamma.div(denom)
            bias = beta - A.mul(mu)
            weight = (weight.transpose(0, -1).mul_(A)).transpose(0, -1)
            # assign to the new conv
            temp_conv.weight.data.copy_(weight)
            temp_conv.bias = torch.nn.Parameter(bias)
            stack.append(temp_conv)
        else:
            stack.append(m)
    if flag:
        return nn.Sequential(*stack)
    else:
        return block

def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])
    return model