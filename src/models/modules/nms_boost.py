# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/4/6 9:02
# @Author : liumin
# @File : nms_boost.py

import numpy as np


def nms(dets, thresh):
    x1 = dets[:, 0]  # pred bbox top_x
    y1 = dets[:, 1]  # pred bbox top_y
    x2 = dets[:, 2]  # pred bbox bottom_x
    y2 = dets[:, 3]  # pred bbox bottom_y
    scores = dets[:, 4]  # pred bbox cls score

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # pred bbox areas
    order = scores.argsort()[::-1]  # 对pred bbox按置信度做降序排序

    keep = []
    while order.size > 0:
        i = order[0]  # top-1 score pred bbox
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        IoU = inter / (areas[i] + areas[order[1:]] - inter)  # IoU计算

        inds = np.where(IoU <= thresh)[0]  # 保留非冗余bbox
        order = order[inds + 1]

    return keep  # 最终NMS结果返回


import numpy as np


def soft_nms(dets, method, thresh=0.001, Nt=0.1, sigma=0.5):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (y2 - y1 + 1.) * (x2 - x1 + 1.)
    orders = scores.argsort()[::-1]
    keep = []

    while orders.size > 0:
        i = orders[0]
        keep.append(i)
        for j in orders[1:]:
            xx1 = np.maximum(x1[i], x1[j])
            yy1 = np.maximum(y1[i], y1[j])
            xx2 = np.minimum(x2[i], x2[j])
            yy2 = np.minimum(y2[i], y2[j])

            w = np.maximum(xx2 - xx1 + 1., 0.)
            h = np.maximum(yy2 - yy1 + 1., 0.)
            inter = w * h
            IoU = inter / (areas[i] + areas[j] - inter)

            if method == 1:  # linear
                if IoU > Nt:
                    weight = 1 - IoU
                else:
                    weight = 1
            elif method == 2:  # gaussian
                weight = np.exp(-(IoU * IoU) / sigma)
            else:  # original NMS
                if IoU > Nt:
                    weight = 0
                else:
                    weight = 1
            scores[j] = weight * scores[j]

            if scores[j] < thresh:
                orders = np.delete(orders, np.where(orders == j))

        orders = np.delete(orders, 0)

    return keep