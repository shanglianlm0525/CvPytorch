# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/13 15:28
# @Author : liumin
# @File : t_nanodet_coco.py

import torch
import torch.nn as nn
import torchvision
import os
import sys
import numpy as np

from src.evaluator.eval_coco import CocoEvaluator
from tqdm import tqdm

'''
    print('result_list', result_list)
    preds = {}
    warp_matrix = meta['warp_matrix'][0] if isinstance(meta['warp_matrix'], list) else meta['warp_matrix']
    warp_matrix = warp_matrix.cpu().numpy() \
        if isinstance(warp_matrix, torch.Tensor) else warp_matrix
    img_height = meta['height'].cpu().numpy() \
        if isinstance(meta['height'], torch.Tensor) else meta['height']
    img_width = meta['width'].cpu().numpy() \
        if isinstance(meta['width'], torch.Tensor) else meta['width']
    for result in result_list:
        det_bboxes, det_labels = result
        det_bboxes = det_bboxes.cpu().numpy()
        det_bboxes[:, :4] = warp_boxes(det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height)
        classes = det_labels.cpu().numpy()
        for i in range(self.num_classes):
            inds = (classes == i)
            preds[i] = np.concatenate([
                det_bboxes[inds, :4].astype(np.float32),
                det_bboxes[inds, 4:5].astype(np.float32)], axis=1).tolist()
    return preds
'''

def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes

category2id = torch.load('/home/lmin/pythonCode/scripts/weights/nanodet/category2id.pth')
id2category = torch.load('/home/lmin/pythonCode/scripts/weights/nanodet/id2category.pth')

root_path = '/home/lmin/pythonCode/scripts/weights/nanodet'

dataset = torch.load(os.path.join(root_path, 'datasets.pth'))
iou_types = ["bbox"]
cocoEvaluator = CocoEvaluator(dataset, iou_types)

for idx in tqdm(range(5000)):
    dets = torch.load(os.path.join(root_path, str(idx)+'_dets.pth'))
    meta = torch.load(os.path.join(root_path, str(idx)+'_meta.pth'))
    result_list = torch.load(os.path.join(root_path, str(idx)+'_result_list.pth'))

    outs = []
    warp_matrix = meta['warp_matrix'][0]
    width = meta['img_info']['width'].cpu().numpy()
    height = meta['img_info']['height'].cpu().numpy()

    preds = {}
    for o in result_list:
        o_bboxes, o_labels = o
        o_bboxes = o_bboxes.cpu().numpy()

        o_bboxes[:, :4] = warp_boxes(o_bboxes[:, :4], np.linalg.inv(warp_matrix), width, height)
        classes = o_labels.cpu().numpy()
        for i in range(80):
            inds = (classes == i)
            preds[i] = np.concatenate([
                o_bboxes[inds, :4].astype(np.float32),
                o_bboxes[inds, 4:5].astype(np.float32)], axis=1).tolist()

        outs.append({"boxes": torch.tensor(o_bboxes[:, :4]).cuda(), "labels": torch.tensor(o_labels).cuda(), "scores": torch.tensor(o_bboxes[:, 4]).cuda()})

    targets = {}
    targets["image_id"] = meta['img_info']['id']
    cocoEvaluator.update([targets], outs)

cocoEvaluator.evaluate()
