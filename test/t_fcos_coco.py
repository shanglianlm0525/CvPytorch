# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/14 18:20
# @Author : liumin
# @File : t_fcos_coco.py

import torch
import torch.nn as nn
import torchvision
import os
import sys
import numpy as np

from src.evaluator.eval_coco import CocoEvaluator
from tqdm import tqdm

root_path = '/home/lmin/pythonCode/scripts/weights'

dataset = torch.load(os.path.join(root_path, 'datasets.pth'))
iou_types = ["bbox"]
cocoEvaluator = CocoEvaluator(dataset, iou_types)


category2id = torch.load('/home/lmin/pythonCode/scripts/weights/category2id.pth')
id2category = torch.load('/home/lmin/pythonCode/scripts/weights/id2category.pth')

'''
{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
{1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59, 55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}

'''

for idx in tqdm(range(4952)):
    image_id = torch.load(os.path.join(root_path, str(idx) + '_image_id.pth'))
    scores = torch.load(os.path.join(root_path,  str(idx) + '_scores.pth'))
    labels = torch.load(os.path.join(root_path,  str(idx) + '_labels.pth'))
    boxes = torch.load(os.path.join(root_path,  str(idx) + '_boxes.pth'))
    scales = torch.load(os.path.join(root_path,  str(idx) + '_scales.pth'))

    # boxes[:, :, 2] -= boxes[:, :, 0]
    # boxes[:, :, 3] -= boxes[:, :, 1]

    threshold = 0.05
    outputs = []
    for boxes, labels, scores in zip(boxes, labels, scores):

        # labels = torch.Tensor([torch.tensor(id2category[l.item()]).cuda() for l in labels])

        # scores are sorted, so we can break
        keep = scores > threshold
        outputs.append({"boxes": boxes[keep]/scales, "labels": labels[keep]-1,
                        "scores": scores[keep]})
    targets = {}
    targets["image_id"] = torch.tensor(image_id)

    cocoEvaluator.update([targets], outputs)
    # assert 1==2

cocoEvaluator.evaluate()


print()