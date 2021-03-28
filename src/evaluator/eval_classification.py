# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/26 8:57
# @Author : liumin
# @File : eval_classification.py

import numpy as np
import torch
from .base_evaluator import BaseEvaluator


class ClassificationEvaluator(BaseEvaluator):
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_class = dataset.num_classes
        self.dictionary = dataset.dictionary
        self.gt_labels = []
        self.pred_labels = []
        self.count = 0

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

    def evaluate(self):
        if self.count < 1:
            return None
        performances = self.Accuracy()
        performances['performance'] = performances['mAcc'] = self.Mean_Accuracy()
        return performances

    def update(self, gt_label, pred_label):
        assert gt_label.shape == pred_label.shape
        gt_label = gt_label.data.cpu().tolist()
        pred_label = pred_label.data.cpu().tolist()
        self.gt_labels.extend(gt_label)
        self.pred_labels.extend(pred_label)
        self.count = self.count + 1

    def reset(self):
        self.gt_labels = []
        self.pred_labels = []
