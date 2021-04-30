# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/23 13:14
# @Author : liumin
# @File : ptq_momic.py
import glob
import os

import cv2
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

        # ====> step 1 : find the max value.



        # ====> step 2 : generate the histogram_interval.


        #  ====> step 3 : generate the histogram.



        # ====> step 4 : using kld to find the best threshold value.


model = resnet18(True)
model.eval()
# print(model)

'''
for n, p in model.named_parameters():
    print(n, p)
'''

fuse_model = fuse_bn_recursively(copy.deepcopy(model))
fuse_model.eval()


img_dir = '/home/lmin/data/aDALI/train/ants'
paths = glob.glob(os.path.join(img_dir, '*.jpg'))
img_list = [cv2.imread(path) for path in paths]

for n, p in fuse_model.named_parameters(): # dict(model.named_parameters()).items():
    # print(n, p.shape, p)
    scales = get_conv_weight_blob_scales(p)


quantize_the_parameters(img_list)



import torchvision.models.quantization as models

# You will need the number of filters in the `fc` for future use.
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_fe = models.resnet18(pretrained=True, progress=False, quantize=True)
num_ftrs = model_fe.fc.in_features


for n, p in fuse_model.named_parameters():
    print(n, p.shape, p)