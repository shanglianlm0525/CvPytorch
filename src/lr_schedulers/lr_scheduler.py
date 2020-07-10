# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 13:01
# @Author : liumin
# @File : lr_scheduler.py
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

def parser_lr_scheduler(cfg, optimizer):
    if cfg.LR_SCHEDULER.TYPE == "MultiStepLR":
        lr_scheduler_ft = lr_scheduler.MultiStepLR(
            optimizer, cfg.LR_SCHEDULER.MILESTONES, gamma=cfg.LR_SCHEDULER.GAMMA or 0.1
        )
    elif cfg.LR_SCHEDULER.TYPE == "ReduceLROnPlateau":
        lr_scheduler_ft = lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=cfg.LR_SCHEDULER.GAMMA or 0.1, patience=cfg.LR_SCHEDULER.PATIENCE
        )
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

'''
net = modelsT.shufflenet_v2_x0_5(pretrained=True)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
warmup_scheduler = WarmUpLR(optimizer, 3224 * 5)
print(warmup_scheduler)
'''