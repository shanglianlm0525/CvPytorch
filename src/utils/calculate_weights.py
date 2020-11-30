# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/21 13:28
# @Author : liumin
# @File : calculate_weights.py

import os

import torch
from tqdm import tqdm
import numpy as np


'''
    compute segmentation classes weights
    https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/calculate_weights.py
'''
def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    for sample in dataloader:
        y = sample['target']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    return ret


'''
    compute detection classes weights
    https://github.com/ultralytics/yolov5/blob/master/utils/general.py
'''
def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)
