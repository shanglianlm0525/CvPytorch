# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/18 10:17
# @Author : liumin
# @File : CustomDataLoader.py

import torch
from torch.utils.data.dataloader import DataLoader


class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', self._RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    class _RepeatSampler(object):
        """ Sampler that repeats forever.

        Args:
            sampler (Sampler)
        """

        def __init__(self, sampler):
            self.sampler = sampler

        def __iter__(self):
            while True:
                yield from iter(self.sampler)