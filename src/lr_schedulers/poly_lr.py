# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/4 8:36
# @Author : liumin
# @File : poly_lr.py

from torch.optim.lr_scheduler import _LRScheduler, StepLR


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        '''
        factor = pow(1 - self.last_epoch / self.max_iters, self.power)
        return [(base_lr) * factor for base_lr in self.base_lrs]
        '''
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]

