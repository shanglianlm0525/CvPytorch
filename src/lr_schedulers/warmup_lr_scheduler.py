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
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR



def AssemblyParams(f):
    @wraps(f)
    def info(*args, **kwargs):
        cfg = kwargs['cfg']
        assert cfg is not None, 'cfg is not None'
        assert isinstance(cfg.WARMUP.NAME, str), 'cfg.WARMUP.ITERS must be str type'
        assert cfg.WARMUP.NAME, 'cfg.WARMUP.NAME must in ["linear","exponent","sine"]'
        assert isinstance(cfg.WARMUP.ITERS, int), 'cfg.WARMUP.ITERS must be int type'
        args[0].warmup_method = cfg.WARMUP.NAME
        args[0].warmup_iters = cfg.WARMUP.ITERS
        f(*args, **kwargs)
    return info

def _get_warmup_factor_at_iter(method: str, warmup_iters: int, iter: int):
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
    if iter >= warmup_iters:
        return 1.0

    if method == "linear":
        return (iter+1) / warmup_iters
    elif method == "exponent":
        return 1.0 - math.exp(-(iter+1) / warmup_iters)
    else:
        return 1.0



class _WarmupLRScheduler(_LRScheduler):
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

    @AssemblyParams
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1, cfg=None):
        self.step_size = step_size
        self.gamma = gamma
        super(WarmupStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.warmup_iters, iter)
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

    @AssemblyParams
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, cfg=None):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.warmup_iters, iter)
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

    @AssemblyParams
    def __init__(self, optimizer, gamma, last_epoch=-1, cfg=None):
        self.gamma = gamma
        super(WarmupExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.warmup_iters, iter)
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
    @AssemblyParams
    def __init__(self, optimizer,T_max, eta_min=0, last_epoch=-1, cfg=None):
        self.T_max = T_max
        self.eta_min = eta_min
        '''
        assert cfg is not None, 'cfg is not None'
        assert isinstance(cfg.WARMUP.NAME, str), 'cfg.WARMUP.ITERS must be str type'
        assert cfg.WARMUP.NAME, 'cfg.WARMUP.NAME must in ["linear","exponent","sine"]'
        assert isinstance(cfg.WARMUP.ITERS, int), 'cfg.WARMUP.ITERS must be int type'
        self.warmup_method = cfg.WARMUP.NAME
        self.warmup_iters = cfg.WARMUP.ITERS
        '''
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, iter):
        warmup_factor = _get_warmup_factor_at_iter(self.warmup_method, self.warmup_iters, iter)
        if iter <= self.warmup_iters:
            return [self.eta_min + warmup_factor*(base_lr - self.eta_min) for base_lr in self.base_lrs]

        return [self.eta_min + (base_lr - self.eta_min) *
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]



class WarmupReduceLROnPlateau():
    """Reduce learning rate when a metric has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This scheduler reads a metrics
        quantity and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.

        Example:
            >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
            >>> for epoch in range(10):
            >>>     train(...)
            >>>     val_loss = validate(...)
            >>>     # Note that step should be called after validate()
            >>>     scheduler.step(val_loss)
        """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)