# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/22 15:44
# @Author : liumin
# @File : warmup.py


def get_warmup_lr(cur_iters, cfg):
    if cfg.WARMUP.NAME == 'constant':
        warmup_lr = cfg.INIT_LR * cfg.WARMUP.FACTOR
    elif cfg.WARMUP.NAME == 'linear':
        k = (1 - cur_iters / cfg.WARMUP.ITERS) * (1 - cfg.WARMUP.FACTOR)
        warmup_lr = cfg.INIT_LR * (1 - k)
    elif cfg.WARMUP.NAME == 'exp':
        k = cfg.WARMUP.FACTOR ** (1 - cur_iters / cfg.WARMUP.ITERS)
        warmup_lr = cfg.INIT_LR * k
    else:
        raise Exception('Unsupported warm up type!')
    return warmup_lr