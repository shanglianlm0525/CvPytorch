# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/16 15:38
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from torch.optim import SGD, Adam, RMSprop

__all__ = ['SGD','Adam','RMSprop']


def build_optimizer(cfg, model):
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
        '''torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)'''
        optimizer = SGD(_params)
    elif cfg.OPTIMIZER.TYPE == "Adam":
        '''torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)'''
        optimizer = Adam(_params)
    elif cfg.OPTIMIZER.TYPE == 'RMSprop':
        '''torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)'''
        optimizer = RMSprop(_params)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(cfg.OPTIMIZER.TYPE))

    return optimizer