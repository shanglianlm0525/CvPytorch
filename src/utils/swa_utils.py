# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/31 19:01
# @Author : liumin
# @File : swa_utils.py

import torch
from torch.optim.swa_utils import AveragedModel, SWALR
from ..optimizers import build_optimizer


def swa(self, scaler, model, datasets, dataloaders, timer, lossLogger, performanceLogger, cfg):
    swa_model = AveragedModel(model)
    swa_optimizer = build_optimizer(cfg, model)
    swa_scheduler = SWALR(swa_optimizer, swa_lr=cfg.SWA.LR, anneal_epochs=cfg.SWA.EPOCHS, anneal_strategy=cfg.SWA.NAME)

    for epoch in range(cfg.SWA.EPOCHS):
        if cfg.distributed:
            dataloaders['train'].sampler.set_epoch(epoch)
        self.train_epoch(scaler, epoch, cfg.SWA.EPOCHS, model, datasets['train'], dataloaders['train'], swa_optimizer, timer, lossLogger, performanceLogger)
        swa_model.update_parameters(model)
        swa_scheduler.step()

    # Update bn statistics for the swa_model at the end
    customize_update_bn(dataloaders['train'], swa_model)

    acc, perf_rst = self.val_epoch(0, 0, swa_model, datasets['val'], dataloaders['val'], timer, lossLogger, performanceLogger)
    return swa_model, acc, perf_rst


def customize_update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for sample in loader:
        imgs, targets = sample['image'], sample['target']
        imgs = list(img.cuda() for img in imgs) if isinstance(imgs, list) else imgs.cuda()
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        if device is not None:
            imgs = imgs.to(device)

        model(imgs)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)