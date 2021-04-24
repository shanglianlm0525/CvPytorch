# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/23 13:14
# @Author : liumin
# @File : ptq_momic.py

import torch
import torch.nn as nn
import copy
from torchvision.models import resnet18
from bn_fusion import fuse_bn_recursively

torch.set_grad_enabled(False)

def get_conv_weight_blob_scales(_params, flag=127):
    # just support Convolution and InnerProduct ignore ConvolutionDepthWise
    params = copy.deepcopy(_params)
    max_value = torch.amax(torch.flatten(torch.abs(params), 1), 1)
    return flag / max_value


def quantize_the_parameters(img_list):
    for img in enumerate(img_list):
        pass


model = resnet18(True)
model.eval()
# print(model)

'''
for n, p in model.named_parameters():
    print(n, p)
'''

fuse_model = fuse_bn_recursively(copy.deepcopy(model))
fuse_model.eval()

for n, p in fuse_model.named_parameters(): # dict(model.named_parameters()).items():
    # print(n, p.shape, p)
    get_conv_weight_blob_scales(p)

