# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/2 18:25
# @Author : liumin
# @File : ious.py

import math
import numpy as np


def IoU(box1, box2):
    # 计算相交区域的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    interArea = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    unionArea = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - interArea

    # 计算IOU
    IoU = interArea/unionArea

    return IoU


def GIoU(box1, box2):
    # 计算最小包庇面积
    cArea = (max(box1[0], box1[2], box2[0], box2[2]) - min(box1[0], box1[2], box2[0], box2[2])) * \
            (max(box1[1], box1[3], box2[1], box2[3]) - min(box1[1], box1[3], box2[1], box2[3]))

    # 计算相交区域的宽高
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    # 计算交集、并集面积
    interArea = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    unionArea = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                (box2[2] - box2[0]) * (box2[3] - box2[1]) - interArea

    # 计算IOU
    IoU = interArea/unionArea

    # 计算空白部分占比
    endArea = (cArea - unionArea)/cArea
    GIoU = IoU - endArea

    return GIoU


def DIoU(box1, box2):
    # 计算对角线长度
    c = np.sqrt((max(box1[0], box1[2], box2[0], box2[2]) - min(box1[0], box1[2], box2[0], box2[2]))**2 + \
                (max(box1[1], box1[3], box2[1], box2[3]) - min(box1[1], box1[3], box2[1], box2[3]))**2)

    # 计算中心点间距
    point_1 = ((box1[0] + box1[2])/2, (box1[1] + box1[3])/2)
    point_2 = ((box2[0] + box2[2])/2, (box2[1] + box2[3])/2)
    d = np.sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)

    # 计算IOU
    iou = IoU(box1, box2)

    # 计算空白部分占比
    lens = d**2 / c**2
    diou = iou - lens

    return diou


def CIoU(box1, box2):

    iou = IoU(box1, box2)
    diou = DIoU(box1, box2)

    v = 4/math.pi**2 * (math.atan((box1[2] - box1[0])/(box1[3] - box1[1])) - math.atan((box2[2] - box2[0])/(box2[3] - box2[1])))**2
    alpha = v / (1-iou) + v

    ciou = diou - alpha * v

    return ciou