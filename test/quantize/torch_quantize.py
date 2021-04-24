# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/22 10:58
# @Author : liumin
# @File : torch_quantize.py

import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet18
import torch.quantization


device = torch.device("cuda:0")

model = resnet18(pretrained=True, progress=False)
print(model)

model.to(device)
model.eval()

modules_to_fuse = [['conv1', 'bn1'],
                   ['layer1.0.conv1', 'layer1.0.bn1'],
                   ['layer1.0.conv2', 'layer1.0.bn2'],
                   ['layer1.1.conv1', 'layer1.1.bn1'],
                   ['layer1.1.conv2', 'layer1.1.bn2'],
                   ['layer2.0.conv1', 'layer2.0.bn1'],
                   ['layer2.0.conv2', 'layer2.0.bn2'],
                   ['layer2.0.downsample.0', 'layer2.0.downsample.1'],
                   ['layer2.1.conv1', 'layer2.1.bn1'],
                   ['layer2.1.conv2', 'layer2.1.bn2'],
                   ['layer3.0.conv1', 'layer3.0.bn1'],
                   ['layer3.0.conv2', 'layer3.0.bn2'],
                   ['layer3.0.downsample.0', 'layer3.0.downsample.1'],
                   ['layer3.1.conv1', 'layer3.1.bn1'],
                   ['layer3.1.conv2', 'layer3.1.bn2'],
                   ['layer4.0.conv1', 'layer4.0.bn1'],
                   ['layer4.0.conv2', 'layer4.0.bn2'],
                   ['layer4.0.downsample.0', 'layer4.0.downsample.1'],
                   ['layer4.1.conv1', 'layer4.1.bn1'],
                   ['layer4.1.conv2', 'layer4.1.bn2']]
model = torch.quantization.fuse_modules(model, modules_to_fuse)
if fbgemm:
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
else:
    model.qconfig = torch.quantization.default_qconfig
torch.quantization.prepare(model, inplace=True)
model.eval()
with torch.no_grad():
    for data, target in train_loader:
        model(data)
torch.quantization.convert(model, inplace=True)