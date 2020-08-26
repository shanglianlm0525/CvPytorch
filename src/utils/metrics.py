# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/29 13:22
# @Author : liumin
# @File : metrics.py
from collections import defaultdict
import numpy as np
import torch


class AverageMeter2:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


class LossMeter:
    def __init__(self):
        self.clear()

    def __add__(self, entries):
        _, entry = next(iter(entries.items()))
        if isinstance(entry, np.ndarray) or isinstance(entry, list):
            [self.meters[n].extend(l) for n, l in entries.items()]
        elif isinstance(entry, torch.Tensor):
            [self.meters[n].append(l.detach().cpu().numpy()) for n, l in entries.items()]
        elif isinstance(entry, float):
            [self.meters[n].append(l) for n, l in entries.items()]
        else:
            raise TypeError
        self.cnt += 1

    def average(self):
        return {k: np.average(v) for k, v in self.meters.items()}

    def clear(self):
        self.meters = defaultdict(list)
        self.cnt = 0

    def __len__(self):
        return self.cnt

    def __str__(self):
        return str(self.meters)

class PerfMeter():
    def __init__(self):
        self.clear()

    def __add__(self, entries):
        _, entry = next(iter(entries.items()))
        if isinstance(entry, np.ndarray) or isinstance(entry, list):
            [self.meters[n].extend(l) for n, l in entries.items()]
        elif isinstance(entry, torch.Tensor):
            [self.meters[n].append(l.detach().cpu().numpy()) for n, l in entries.items()]
        elif isinstance(entry, float):
            [self.meters[n].append(l) for n, l in entries.items()]
        else:
            raise TypeError
        self.cnt += 1

    def average(self):
        return {k: np.average(v) for k, v in self.meters.items()}

    def clear(self):
        self.meters = defaultdict(list)
        self.cnt = 0

    def __len__(self):
        return self.cnt

    def __str__(self):
        return str(self.meters)

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num=1):
        if num>0:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


class BaseMeter(object):
    def __init__(self, dict):
        self.dict = dict
        self.predix = ''
        self.reset()

    def reset(self):
        self.meters = defaultdict()
        self.meters['all'+self.predix] = AverageMeter()
        for d in self.dict:
            for k in d.keys():
                self.meters[k+self.predix] = AverageMeter()
        self.cnt = 0

    def __add__(self, entries):
        _, entry = next(iter(entries.items()))
        if isinstance(entry, np.ndarray) or isinstance(entry, list):
            [self.meters[k].update(v, entries[k + '_num'].detach().cpu().item()) for k, v in entries.items() if not k.endswith('_num')]
        elif isinstance(entry, torch.Tensor):
            if not entry.dim():
                [self.meters[k].update(v.detach().cpu().item(), entries[k + '_num'].detach().cpu().item()) for k, v in entries.items() if not k.endswith('_num')]
            else:
                [self.meters[k].update(v[i].detach().cpu().item(),entries[k+'_num'][i].detach().cpu().item()) for k, v in entries.items() if not k.endswith('_num') for i in range(len(v))]
        else:
            raise TypeError
        self.cnt += 1

    def average(self, batch_size):
        return {k: v.avg/batch_size for k, v in self.meters.items()}

    def clear(self):
        self.reset()

    def __str__(self):
        return str(self.meters)

class LossMeter2(BaseMeter):
    def __init__(self, dict):
        super(BaseMeter, self).__init__()
        self.dict = dict
        self.predix = '_loss'
        self.reset()

class PerformanceMeter2(BaseMeter):
    def __init__(self, dict):
        super(BaseMeter, self).__init__()
        self.dict = dict
        self.predix = '_perf'
        self.reset()


class SegmentationEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)