# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/30 16:16
# @Author : liumin
# @File : einops_ops.py

'''
    PyTorch 70.einops：优雅地操作张量维度
    https://zhuanlan.zhihu.com/p/342675997
'''

from einops import rearrange, reduce, repeat
# stacking, reshape, transposition, squeeze/unsqueeze, repeat, tile, concatenate, view



class AdaptiveAvgPool2d(object):
    def forward(self, x):
        return reduce(x, 'b c h w -> b c', reduction='mean')



if __name__ == "__main__":
    import torch


    x = torch.randn([10, 32, 100, 200])