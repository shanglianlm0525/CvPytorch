# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/24 16:45
# @Author : liumin
# @File : recoder.py
from collections import defaultdict

import torch
import numpy as np

class LossRecoder(object):
    def __init__(self, dict,predix='_loss'):
        self.dict = dict
        self.predix = predix
        self.recoder = defaultdict(list)
        self.recoder['all'+self.predix] = []
        for d in self.dict:
            for k in d.keys():
                self.recoder[k+self.predix] = []
        self.iter_num = 0

    def __add__(self, losses):
        _, ls = next(iter(losses.items()))
        if isinstance(ls, np.ndarray) or isinstance(ls, list):
            [self.recoder[n].extend(l) for n, l in losses.items()]
        elif isinstance(ls, torch.Tensor):
            [self.recoder[n].append(l.detach().cpu().tolist()) for n, l in losses.items()]
        else:
            raise TypeError
        self.iter_num += 1

    def average(self):
        return {n: np.average(l) for n, l in self.recoder.items()}

    def summation(self):
        return {n: np.sum(l) for n, l in self.recoder.items()}

    def clear(self):
        self.recoder = defaultdict(list)
        self.recoder['all' + self.predix] = []
        for d in self.dict:
            for k in d.keys():
                self.recoder[k + self.predix] = []
        self.iter_num = 0

    def __str__(self):
        return str(self.recoder)

class PerformanceRecoder(object):
    def __init__(self, dict,predix='_perf'):
        self.dict = dict
        self.predix = predix
        self.recoder = defaultdict(list)
        self.recoder['all'+self.predix] = []
        for d in self.dict:
            for k in d.keys():
                self.recoder[k+self.predix] = []
        self.iter_num = 0

    def __add__(self, losses):
        _, ls = next(iter(losses.items()))
        if isinstance(ls, np.ndarray) or isinstance(ls, list):
            [self.recoder[n].extend(l) for n, l in losses.items()]
        elif isinstance(ls, torch.Tensor):
            [self.recoder[n].append(l.detach().cpu().tolist()) for n, l in losses.items()]
        else:
            raise TypeError
        self.iter_num += 1

    def average(self):
        return {n: np.average(l) for n, l in self.recoder.items()}

    def summation(self):
        return {n: np.sum(l) for n, l in self.recoder.items()}

    def clear(self):
        self.recoder = defaultdict(list)
        self.recoder['all' + self.predix] = []
        for d in self.dict:
            for k in d.keys():
                self.recoder[k + self.predix] = []
        self.iter_num = 0

    def __str__(self):
        return str(self.recoder)

class LossRecoder2:
    def __init__(self):
        self.losses = defaultdict(list)
        self.it = 0

    def put(self, losses):
        _, ls = next(iter(losses.items()))
        if isinstance(ls, np.ndarray) or isinstance(ls, list):
            [self.losses[n].extend(l) for n, l in losses.items()]
        elif isinstance(ls, torch.Tensor):
            for n, l in losses.items():
                print(n, l,l.detach().cpu().numpy())
            [self.losses[n].extend(l.detach().cpu().numpy()) for n, l in losses.items()]
        else:
            raise TypeError
        self.it += 1

    def get(self):
        return {n: np.average(l) for n,  l in self.losses.items()}

    def clear(self):
        self.losses = defaultdict(list)
        self.it = 0

class PerformanceRecoder2:
    def __init__(self):
        self.performances = []
        self.it = 0

    def put(self, performances):
        for i, p in enumerate(performances):
            perf = defaultdict(list)
            _, pf = next(iter(p.items()))
            if isinstance(pf, np.ndarray) or isinstance(pf, list):
                [perf[n].extend(l) for n, l in p.items()]
            elif isinstance(pf, torch.Tensor):
                [perf[n].extend(l.detach().cpu().numpy()) for n, l in p.items()]
            elif isinstance(pf, float):
                [perf[n].append(l) for n, l in p.items()]
            else:
                raise TypeError

            try:
                _classes = set(self.performances[i].keys()).union(set(perf.keys()))
                for k in _classes:
                    self.performances[i][k].extend(perf[k])
            except IndexError:
                self.performances.append(perf)

        self.it += 1

    def get(self):
        return [{n: np.average(l) for n, l in p.items()} for p in self.performances]

    def clear(self):
        self.performances = []
        self.it = 0

class TimeRecoder:
    def __init__(self):
        self.times = []
        self.it = 0

    def put(self, t):
        self.times.append(t)
        self.it += 1

    def get(self):
        return sum(self.times)

    def clear(self):
        self.times = []
        self.it = 0