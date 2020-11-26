# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/26 8:57
# @Author : liumin
# @File : eval_classification.py

import numpy as np
import torch


class ClassificationEvaluator(object):
    def __init__(self, dictionary, iou_thread=0.5):
        self.dictionary = dictionary
        self.gt_labels = []
        self.pred_labels = []

    def Accuracy(self):
        all_acc = {}
        for idx, d in enumerate(self.dictionary):
            for _label, _weight in d.items():
                all_acc[_label] = (np.equal(self.gt_labels, self.pred_labels) & np.equal(self.gt_labels, idx)).sum() / (
                            np.equal(self.gt_labels, idx).sum() + 1e-6)
        return all_acc

    def Mean_Accuracy(self):
        correct = np.equal(self.gt_labels, self.pred_labels).sum()
        accuracy = correct/(len(self.gt_labels)+1e-6)
        return accuracy

    def add_batch(self, gt_label, pred_label):
        assert gt_label.shape == pred_label.shape
        gt_label = gt_label.data.cpu().tolist()
        pred_label = pred_label.data.cpu().tolist()
        self.gt_labels.extend(gt_label)
        self.pred_labels.extend(pred_label)

    def reset(self):
        self.gt_labels = []
        self.pred_labels = []



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res