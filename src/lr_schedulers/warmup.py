# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/22 15:44
# @Author : liumin
# @File : warmup.py


def get_warmup_lr(cur_iters, cfg):
    if cfg.WARMUP.NAME == 'constant':
        warmup_lr = cfg.INIT_LR * cfg.WARMUP.FACTOR
    elif cfg.WARMUP.NAME == 'linear':
        # k = (1 - cur_iters / cfg.WARMUP.ITERS) * (1 - cfg.WARMUP.FACTOR)
        # warmup_lr = cfg.INIT_LR * (1 - k)
        alpha = cur_iters / cfg.WARMUP.ITERS
        warmup_factor = cfg.WARMUP.FACTOR * (1.0 - alpha) + alpha
        warmup_lr = cfg.INIT_LR * warmup_factor
    elif cfg.WARMUP.NAME == 'exp':
        k = cfg.WARMUP.FACTOR ** (1 - cur_iters / cfg.WARMUP.ITERS)
        warmup_lr = cfg.INIT_LR * k
    else:
        raise Exception('Unsupported warm up type!')
    return warmup_lr


'''
def poly_lr_scheduler(current_iter, total_iters,warmup_iters,warmup_factor,p=0.9):
    lr=(1 - current_iter / total_iters) ** p
    if current_iter<warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr
def exp_lr_scheduler(current_iter, total_iters,warmup_iters,warmup_factor,beta):
    lr=beta**(current_iter/total_iters)
    if current_iter<warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr

def cosine_lr_scheduler(current_iter, total_iters,warmup_iters,warmup_factor,min_lr=0):
    lr = 0.5 * (1 + math.cos(current_iter / total_iters * math.pi))
    lr=lr*(1-min_lr)+min_lr
    if current_iter<warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr

def step_lr_scheduler(current_iter, total_iters,warmup_iters,warmup_factor):
    # following https://github.com/facebookresearch/detectron2/blob/main/configs/new_baselines/mask_rcnn_R_50_FPN_100ep_LSJ.py
    if current_iter/total_iters < 0.89:
        lr=1
    elif current_iter/total_iters < 0.96:
        lr=0.1
    else:
        lr=0.01
    if current_iter<warmup_iters:
        alpha=warmup_factor+(1-warmup_factor)*(current_iter/warmup_iters)
        lr*=alpha
    return lr
'''