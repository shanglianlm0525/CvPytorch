# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/17 20:51
# @Author : liumin
# @File : psp_neck.py

from src.models.necks.seg.base_seg_neck import BaseSegNeck


class PSPNeck(BaseSegNeck):
    def forward(self, x):
        return x[-1], x[:-1]