# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/28 17:32
# @Author : liumin
# @File : prefetch_dataLoader.py

import torch
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class PrefetchDataLoader(DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



class DataPrefetcher(object):
    '''
        prefetcher = DataPrefetcher(train_loader, device=self.device)
        batch = prefetcher.next()
        iter_id = 0
        while batch is not None:
            iter_id += 1
            if iter_id >= num_iters:
                break
            run_step()
            batch = prefetcher.next()
    '''
    def __init__(self, loader, device):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.device = device

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            self.preload()
            count += 1
            yield input, target