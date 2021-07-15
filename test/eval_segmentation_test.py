# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/3 17:54
# @Author : liumin
# @File : eval_segmentation_test.py

import os
import sys
import numpy as np


class SegmentationEvaluator():
    def __init__(self, num_classes):
        self.num_class = num_classes
        self.confusion_matrix = np.ones((self.num_class,)*2)
        self.count = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = np.nanmean(Acc)
        return mAcc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        # print('Class IoU', MIoU)
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

    def evaluate(self):
        if self.count < 1:
            return None
        performances = {}
        performances['Acc'] = self.Pixel_Accuracy()
        performances['mAcc'] = self.Mean_Pixel_Accuracy()
        performances['mIoU'] = self.Mean_Intersection_over_Union()
        performances['FWIoU'] = self.Frequency_Weighted_Intersection_over_Union()
        performances['performance'] = performances['mIoU']
        return performances

    def update(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        # gt_image = gt_image.data.cpu().numpy()
        # pre_image = pre_image.data.cpu().numpy()
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.count += 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.count = 0


segmentationEvaluator = SegmentationEvaluator(num_classes=19)


for i in range(125):
    preds = np.load('/home/lmin/pythonCode/CvPytorch/test_data/' + str(i) + '_preds.npy')
    targets = np.load('/home/lmin/pythonCode/CvPytorch/test_data/' + str(i) + '_targets.npy')

    segmentationEvaluator.update(targets, preds)

# {'Acc': 0.9524840160046874, 'mAcc': 0.8005588202083923, 'mIoU': 0.7206716124673626, 'FWIoU': 0.9127624218105044, 'performance': 0.7206716124673626}
performances = segmentationEvaluator.evaluate()
print(performances)