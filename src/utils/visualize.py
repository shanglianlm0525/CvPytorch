# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/22 11:01
# @Author : liumin
# @File : visualize.py

import torch
from thop import profile

@torch.no_grad()
def show_flops_params(model, device, input_shape=[1, 3, 1024, 2048], logger=None):
    #summary(model, tuple(input_shape[1:]), device=device)
    input = torch.randn(*input_shape).to(torch.device(device))
    flops, params = profile(model, inputs=(input,), verbose=False)

    if logger is not None:
        logger.info('{} flops: {:.3f}G input shape is {}, params: {:.3f}M'.format(
            model.__class__.__name__, flops / 1000000000, input_shape[1:], params / 1000000))
