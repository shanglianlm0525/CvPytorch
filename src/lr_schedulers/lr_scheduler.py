# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 13:01
# @Author : liumin
# @File : lr_scheduler.py
import math
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.models as models

def parser_lr_scheduler(cfg, optimizer):
    if cfg.LR_SCHEDULER.TYPE == "MultiStepLR":
        lr_scheduler_ft = lr_scheduler.MultiStepLR(
            optimizer, cfg.LR_SCHEDULER.MILESTONES, gamma=cfg.LR_SCHEDULER.GAMMA or 0.1
        )
    elif cfg.LR_SCHEDULER.TYPE == "ReduceLROnPlateau":
        lr_scheduler_ft = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.LR_SCHEDULER.GAMMA or 0.1, patience=cfg.LR_SCHEDULER.PATIENCE
        )
    elif cfg.LR_SCHEDULER.TYPE == "CosineAnnealingLR":
        """
            \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
        """
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        lf = lambda x: (((1 + math.cos(x * math.pi / cfg.N_MAX_EPOCHS)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        lr_scheduler_ft = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # lr_scheduler_ft = lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.LR_SCHEDULER.PATIENCE, eta_min = 1e-6)
    else:
        raise ValueError("Unsupported lr_scheduler type: {}".format(cfg.LR_SCHEDULER.TYPE))
    return lr_scheduler_ft



class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


net = models.shufflenet_v2_x0_5(pretrained=True)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
warmup_scheduler = WarmUpLR(optimizer, 3224 * 5)
print(warmup_scheduler)
