# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/21 13:28
# @Author : liumin
# @File : calculate_weights.py

import os
from tqdm import tqdm
import numpy as np

def calculate_weigths_labels(dataset, dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    for sample in dataloader:
        y = sample['label']
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