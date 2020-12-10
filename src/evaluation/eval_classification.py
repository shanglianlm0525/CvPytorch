# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/26 8:57
# @Author : liumin
# @File : eval_classification.py

import numpy as np
import torch


class ClassificationEvaluator(object):
    def __init__(self, dictionary):
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

    def compute(self):
        performances = self.Accuracy()
        performances['mAcc'] = self.Mean_Accuracy()
        performances['performance'] = self.Mean_Accuracy()
        return performances

    def add_batch(self, gt_label, pred_label):
        assert gt_label.shape == pred_label.shape
        gt_label = gt_label.data.cpu().tolist()
        pred_label = pred_label.data.cpu().tolist()
        self.gt_labels.extend(gt_label)
        self.pred_labels.extend(pred_label)

    def reset(self):
        self.gt_labels = []
        self.pred_labels = []
