# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/17 9:13
# @Author : liumin
# @File : warmup_lr_scheduler.py
import math
import warnings
from torch._six import inf
from functools import partial, wraps
from bisect import bisect_right

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

def AssemblyParams(f):
    @wraps(f)
    def info(*args, **kwargs):
        cfg = kwargs['cfg']
        assert cfg is not None, 'cfg is not None'
        assert isinstance(cfg.WARMUP.NAME, str), 'cfg.WARMUP.ITERS must be str type'
        assert cfg.WARMUP.NAME, 'cfg.WARMUP.NAME must in ["linear","exponent","sine"]'
        assert isinstance(cfg.WARMUP.ITERS, int), 'cfg.WARMUP.ITERS must be int type'
        assert isinstance(cfg.WARMUP.FACTOR, float), 'cfg.WARMUP.FACTOR must be float type'
        args[0].warmup_method = cfg.WARMUP.NAME
        args[0].warmup_iters = cfg.WARMUP.ITERS
        args[0].warmup_factor = cfg.WARMUP.FACTOR
        f(*args, **kwargs)
    return info


class _WarmupLRScheduler(_LRScheduler):
    @AssemblyParams
    def __init__(self, optimizer, last_epoch=-1, cfg=None):
        super(_WarmupLRScheduler, self).__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(self._step_count)):
            param_group['lr'] = lr

    def get_warmup_factor_at_iter(self, iter):
        """
        Return the learning rate warmup factor at a specific iteration.
        See :paper:`in1k1h` for more details.
        Args:
            method (str): warmup method; either "constant" or "linear".
            iter (int): iteration at which to calculate the warmup factor.
            warmup_iters (int): the number of warmup iterations.
        Returns:
            float: the effective warmup factor at the given iteration.
        """
        if iter >= self.warmup_iters:
            return 1.0

        if self.warmup_method == "constant":
            return self.warmup_factor
        elif self.warmup_method == "linear":
            return (iter + 1) / self.warmup_iters
        elif self.warmup_method == "exponent":
            return 1.0 - math.exp(-(iter + 1) / self.warmup_iters)
        else:
            return 1.0


class WarmupStepLR(_WarmupLRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
        decayed by gamma every step_size epochs. When last_epoch=-1, sets
        initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            step_size (int): Period of learning rate decay.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.

        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 60
            >>> # lr = 0.0005   if 60 <= epoch < 90
            >>> # ...
            >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, cfg=None):
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = self.get_warmup_factor_at_iter(iter)
        if iter <= self.warmup_iters:
            return [self.eta_min + warmup_factor * (base_lr - self.eta_min) for base_lr in self.base_lrs]

        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]



class WarmupMultiStepLR(_WarmupLRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma once the number of epoch reaches one of the milestones. When
        last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            milestones (list): List of epoch indices. Must be increasing.
            gamma (float): Multiplicative factor of learning rate decay.
                Default: 0.1.
            last_epoch (int): The index of last epoch. Default: -1.

        Example:
            >>> # Assuming optimizer uses lr = 0.05 for all groups
            >>> # lr = 0.05     if epoch < 30
            >>> # lr = 0.005    if 30 <= epoch < 80
            >>> # lr = 0.0005   if epoch >= 80
            >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
            >>> for epoch in range(100):
            >>>     train(...)
            >>>     validate(...)
            >>>     scheduler.step()
        """
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, cfg=None):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = self.get_warmup_factor_at_iter(iter)
        if iter <= self.warmup_iters:
            return [self.eta_min + warmup_factor * (base_lr - self.eta_min) for base_lr in self.base_lrs]

        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class WarmupExponentialLR(_WarmupLRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
        by gamma every epoch. When last_epoch=-1, sets initial lr as lr.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            gamma (float): Multiplicative factor of learning rate decay.
            last_epoch (int): The index of last epoch. Default: -1.
        """
    def __init__(self, optimizer, gamma, last_epoch=-1, cfg=None):
        self.gamma = gamma
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = self.get_warmup_factor_at_iter(iter)
        if iter <= self.warmup_iters:
            return [self.eta_min + warmup_factor * (base_lr - self.eta_min) for base_lr in self.base_lrs]

        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]


class WarmupCosineAnnealingLR(_WarmupLRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
        schedule, where :math:`\eta_{max}` is set to the initial lr and
        :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

        .. math::
            \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
            \cos(\frac{T_{cur}}{T_{max}}\pi))

        When last_epoch=-1, sets initial lr as lr.

        It has been proposed in
        `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
        implements the cosine annealing part of SGDR, and not the restarts.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.

        .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
            https://arxiv.org/abs/1608.03983
        """
    def __init__(self, optimizer,T_max, eta_min=0, last_epoch=-1, cfg=None):
        self.T_max = T_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = self.get_warmup_factor_at_iter(iter)
        if iter <= self.warmup_iters:
            return [self.eta_min + warmup_factor*(base_lr - self.eta_min) for base_lr in self.base_lrs]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]
