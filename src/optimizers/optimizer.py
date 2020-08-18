# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 13:01
# @Author : liumin
# @File : optimizer.py
from copy import deepcopy
import torch
import torch.optim as optim

def parser_optimizer(cfg, model):
    # params = [p for p in model.parameters() if p.requires_grad]
    _params = []
    # filter(lambda p: p.requires_grad, model.parameters())
    for n, p in dict(model.named_parameters()).items():
        if p.requires_grad:
            _args = deepcopy(cfg.OPTIMIZER.BIAS_PARAMS if "bias" in n else cfg.OPTIMIZER.WEIGHT_PARAMS)
            _args.pop("data")
            _params += [{"params": [p], "lr": cfg.INIT_LR, **_args}]
            if "bias" in n:
                _params[-1]["lr"] *= cfg.OPTIMIZER.BIAS_LR_MULTIPLIER or 1.0

    if cfg.OPTIMIZER.TYPE == "SGD":
        optimizer = optim.SGD(_params)
    elif cfg.OPTIMIZER.TYPE == "Adam":
        optimizer = optim.Adam(_params)
    elif cfg.OPTIMIZER.TYPE == 'RMSprop':
        optimizer = optim.RMSprop(_params)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(cfg.OPTIMIZER.TYPE))

    return optimizer
