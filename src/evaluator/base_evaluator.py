# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/22 15:08
# @Author : liumin
# @File : base_evaluator.py

class BaseEvaluator(object):
    def __init__(self, name):
        self.name = name

    def evaluate(self):
        raise NotImplementedError

    def update(self, gt_label, pred_label):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError