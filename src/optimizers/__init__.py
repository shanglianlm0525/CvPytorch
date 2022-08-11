# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/5 14:16
# @Author : liumin
# @File : __init__.py
import torch
import torch.nn as nn
from copy import deepcopy
from torch.optim import SGD, Adam, AdamW, RMSprop, Adadelta
from .radam import RAdam
from .ranger import Ranger
from .adabelief import AdaBelief

__all__ = ['SGD','Adam', 'AdamW','Adadelta','RMSprop', 'RAdam', 'Ranger', 'AdaBelief']


def get_current_lr(optimizer):
    return min(g["lr"] for g in optimizer.param_groups)


def build_optimizer(cfg, model):
    '''
    # params = [p for p in model.parameters() if p.requires_grad]
    _params = []
    # filter(lambda p: p.requires_grad, model.parameters())
    for n, p in dict(model.named_parameters()).items():
        if p.requires_grad:
            _args = deepcopy(cfg.OPTIMIZER.BIAS_PARAMS if "bias" in n else cfg.OPTIMIZER.WEIGHT_PARAMS)
            _args.pop("data")
            _params += [{"params": [p], "lr": cfg.BACKBONE_LR if 'backbone' in n and cfg.BACKBONE_LR is not None else cfg.INIT_LR, **_args}]
            if "bias" in n:
                _params[-1]["lr"] *= cfg.OPTIMIZER.BIAS_LR_MULTIPLIER or 1.0
    '''

    # params = [p for p in model.parameters() if p.requires_grad]
    _params = []
    # filter(lambda p: p.requires_grad, model.parameters())
    for n, p in dict(model.named_parameters()).items():
        if p.requires_grad:
            _args = deepcopy(cfg.OPTIMIZER.BIAS_PARAMS if "bias" in n else cfg.OPTIMIZER.WEIGHT_PARAMS)
            _args.pop("data")
            _params += [{"params": [p], "lr": cfg.BACKBONE_LR if 'backbone' in n and cfg.BACKBONE_LR is not None else cfg.INIT_LR, **_args}]
            if "bias" in n:
                _params[-1]["lr"] *= cfg.OPTIMIZER.BIAS_LR_MULTIPLIER or 1.0

    opt_type = cfg.OPTIMIZER.TYPE.lower()

    if opt_type == "sgd":
        '''torch.optim.SGD(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False)'''
        optimizer = SGD(_params)
    elif opt_type == "adam":
        '''torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)'''
        optimizer = Adam(_params)
    elif opt_type == "adamw":
        '''torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)'''
        optimizer = AdamW(_params)
    elif opt_type == "adadelta":
        '''torch.optim.Adadelta(params, lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)'''
        optimizer = Adadelta(_params)
    elif opt_type == 'rmsprop':
        '''torch.optim.RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)'''
        optimizer = RMSprop(_params)
    elif opt_type == 'radam':
        '''optimizer = RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, betas=(0.90, 0.999), eps=1e-08, weight_decay=1e-4)'''
        optimizer = RAdam(_params)
    elif opt_type == 'ranger':
        '''optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01, betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4)'''
        optimizer = Ranger(_params)
    elif opt_type == 'adabelief':
        optimizer = AdaBelief(_params)
    else:
        raise ValueError("Unsupported optimizer type: {}, Expected optimizer method in {} ".format(cfg.OPTIMIZER.TYPE, __all__))

    return optimizer