# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/17 9:57
# @Author : liumin
# @File : freeze.py


def freeze_models(model):
    # Freeze
    freeze = ['', ]  # parameter names to freeze (full or partial)
    if any(freeze):
        for k, v in model.named_parameters():
            if any(x in k for x in freeze):
                print('freezing %s' % k)
                v.requires_grad = False