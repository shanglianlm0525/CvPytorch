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
import bisect
from bisect import bisect_right

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


"""
    Warm-up和Cos设置LR
    https://www.cnblogs.com/wjy-lulu/p/12934333.html
"""

def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
):
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`in1k1h` for more details.
    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).
    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))

class WarmupMultiStepLR():
    def __init__(
        self,
        milestones,
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_iters: int = 50,
        warmup_method: str = "linear",
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}", milestones
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self, iter) :
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, iter, self.warmup_iters, self.warmup_factor
        )
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, iter)
            for base_lr in [0.001]
        ]


class WarmupCosineLR():
    def __init__(
        self,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 50,
        warmup_method: str = "linear",
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method

    def get_lr(self, iter):
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, iter, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * iter / self.max_iters))
            for base_lr in [0.01]]


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
net = models.shufflenet_v2_x0_5(pretrained=True)
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
warmup_scheduler = WarmUpLR(optimizer, 3224 * 5)
print(warmup_scheduler)
'''
