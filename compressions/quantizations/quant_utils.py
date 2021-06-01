# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/26 18:38
# @Author : liumin
# @File : quant_utils.py

import torch
import torch.nn as nn
from compressions.quantizations.quant_modules import QConv2d, QLinear, QIdentity


def get_input_sequences(model, dummy_shape=[1, 3, 224, 224]):
    layer_bn_pairs = []

    def hook(name):
        def func(m, i, o):
            if m in (torch.nn.Conv2d, torch.nn.Linear):
                if not layer_bn_pairs:
                    layer_bn_pairs.append((m, name))
                else:
                    if layer_bn_pairs[-1][0] in (torch.nn.Conv2d, torch.nn.Linear):
                        layer_bn_pairs.pop()
            else:
                layer_bn_pairs.append((m, name))

        return func

    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn(dummy_shape).cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_bn_pairs


def replace_quant_ops(model, w_bit, w_scheme, b_bit, a_bit, a_scheme):
    prev_module = None
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QConv2d(child, w_scheme='mse', w_bit = 8, b_bit=8, a_scheme='mse', a_bit=8)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, torch.nn.Linear):
            new_op = QLinear(child, w_scheme='mse', w_bit = 8, b_bit=8, a_scheme='mse', a_bit=8)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            # prev_module.activation_function = child
            prev_module.activation_function = torch.nn.ReLU()
            setattr(model, child_name, QIdentity())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, QIdentity())
        else:
            replace_quant_ops(child, w_bit, w_scheme, b_bit, a_bit, a_scheme)