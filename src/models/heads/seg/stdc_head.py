# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/6 15:30
# @Author : liumin
# @File : stdc_head.py

import torch
from torch import nn
import torch.nn.functional as F

from src.models.bricks import ConvModule
from src.models.heads.seg.fcn_head import FCNHead



class STDCHead(FCNHead):
    def __init__(self, **kwargs):
        super(STDCHead, self).__init__(**kwargs)





if __name__ == '__main__':
    model = STDCHead(num_classes=19, in_channels=256, channels=256, num_convs=1,dropout_ratio=0.0, is_concat = False)
    print(model)

