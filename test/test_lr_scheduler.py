# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/17 12:49
# @Author : liumin
# @File : test_lr_scheduler.py
import math
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
import itertools
import warnings
import numpy as np
import matplotlib.pyplot as plt

from src.lr_schedulers.warmup_lr_scheduler import WarmupCosineAnnealingLR
from src.utils.config import CommonConfiguration

initial_lr = 0.01
max_epoch = 300

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)

    def forward(self, x):
        pass

net_1 = model()
optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
scheduler_1 = CosineAnnealingLR(optimizer_1, T_max=max_epoch, eta_min=0.02)

cfg = CommonConfiguration.from_yaml('/home/lmin/pythonCode/CvPytorch/conf/hymenoptera.yml')

net_2 = model()
optimizer_2 = torch.optim.Adam(net_2.parameters(), lr=initial_lr)
scheduler_2 = WarmupCosineAnnealingLR(optimizer_2, T_max=max_epoch, eta_min=0.02, cfg=cfg)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


net_3 = model()
optimizer_3 = torch.optim.SGD(net_3.parameters(), lr=0.01, momentum=0.937, nesterov=True)
lf = one_cycle(1, 0.2, max_epoch)  # cosine 1->hyp['lrf']
scheduler_3 = lr_scheduler.LambdaLR(optimizer_3, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)


print("初始化的学习率：", optimizer_1.defaults['lr'])

lr_list1 = []  # 把使用过的lr都保存下来，之后画出它的变化
lr_list2 = []
lr_list3 = []

for epoch in range(0, max_epoch):
    # train
    for iter in range(5):
        optimizer_1.zero_grad()
        optimizer_1.step()
        optimizer_2.zero_grad()
        optimizer_2.step()
        optimizer_3.zero_grad()
        optimizer_3.step()
        print("第%d个epoch的学习率：%f %f %f" % (epoch, optimizer_1.param_groups[0]['lr'], optimizer_2.param_groups[0]['lr'], optimizer_3.param_groups[0]['lr']))
        lr_list1.append(optimizer_1.param_groups[0]['lr'])
        lr_list2.append(optimizer_2.param_groups[0]['lr'])
        lr_list3.append(optimizer_3.param_groups[0]['lr'])
        scheduler_1.step(epoch)
        scheduler_2.step(epoch)
        scheduler_3.step(epoch)

# 画出lr的变化
fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
plt.plot(list(range(500)), lr_list1)
plt.plot(list(range(500)), lr_list2)
plt.plot(list(range(500)), lr_list3)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.show()