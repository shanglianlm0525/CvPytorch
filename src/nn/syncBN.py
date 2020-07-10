# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/2 13:40
# @Author : liumin
# @File : syncBN.py

import torch
import torch.nn as nn

def convertBNtoSyncBN(module, process_group=None):
    '''Recursively replace all BN layers to SyncBN layer.

    Args:
        module[torch.nn.Module]. Network
    '''
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        sync_bn = nn.SyncBatchNorm(module.num_features, module.eps, module.momentum,
                                         module.affine, module.track_running_stats, process_group)
        sync_bn.running_mean = module.running_mean
        sync_bn.running_var = module.running_var
        if module.affine:
            sync_bn.weight = module.weight.clone().detach()
            sync_bn.bias = module.bias.clone().detach()
        return sync_bn
    else:
        for name, child_module in module.named_children():
            setattr(module, name, nn.SyncBatchNorm.convert_sync_batchnorm(child_module, process_group=process_group))
        return module