# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/16 15:38
# @Author : liumin
# @File : __init__.py
import math
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, LambdaLR

__all__ = ['StepLR', 'MultiStepLR', 'ReduceLROnPlateau','CosineAnnealingLR']


def build_lr_scheduler(cfg, optimizer):
    if cfg.LR_SCHEDULER.TYPE == "MultiStepLR":
        lr_scheduler_ft = StepLR(
            optimizer, step_size=cfg.LR_SCHEDULER.PATIENCE, gamma=cfg.LR_SCHEDULER.GAMMA or 0.1
        )
    elif cfg.LR_SCHEDULER.TYPE == "MultiStepLR":
        lr_scheduler_ft = MultiStepLR(
            optimizer, cfg.LR_SCHEDULER.MILESTONES, gamma=cfg.LR_SCHEDULER.GAMMA or 0.1
        )
    elif cfg.LR_SCHEDULER.TYPE == "ReduceLROnPlateau":
        lr_scheduler_ft = ReduceLROnPlateau(
            optimizer, factor=cfg.LR_SCHEDULER.GAMMA or 0.1, patience=cfg.LR_SCHEDULER.PATIENCE
        )
    elif cfg.LR_SCHEDULER.TYPE == "CosineAnnealingLR":
        """
            \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        """
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        lf = lambda x: (((1 + math.cos(x * math.pi / cfg.N_MAX_EPOCHS)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        lr_scheduler_ft = LambdaLR(optimizer, lr_lambda=lf)

        # lr_scheduler_ft = CosineAnnealingLR(optimizer_1, T_max=100, eta_min=0.02)
    else:
        raise ValueError("Unsupported lr_scheduler type: {}".format(cfg.LR_SCHEDULER.TYPE))

    return lr_scheduler_ft