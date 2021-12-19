# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/23 15:22
# @Author : liumin
# @File : eval_yolov5_detection.py

import copy
import torch
import numpy as np
from src.evaluator.base_evaluator import BaseEvaluator


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = ((np.minimum(np.expand_dims(box1[:, 2:], 1), box2[:, 2:]) - np.maximum(np.expand_dims(box1[:, :2], 1),
                                                                           box2[:, :2])).clip(0)).prod(2)
    return inter / (np.expand_dims(area1, 1) + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=np.bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # iouv[0] = 0.01
    x = np.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = np.hstack((np.stack(x, 1), np.expand_dims(iou[x[0], x[1]], 1)))  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(np.int)] = matches[:, 2:3] >= iouv
    return correct


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

class Yolov5Evaluator(BaseEvaluator):
    def __init__(self, dataset=None, iou_thread=0.5):
        # self.dataset = dataset
        # self.num_class = dataset.num_classes
        # self.dictionary = dataset.dictionary
        self.iou_thread = iou_thread

        self.stats = []
        self.iouv = np.linspace(0.5, 0.95, 10)
        self.count = 0


    def update(self, gt_targets, preds):
        for gt_target, pred in zip(gt_targets, preds):
            gt_bbox = gt_target['boxes'].cpu().numpy()
            gt_label = gt_target['labels'].cpu().numpy()

            gt_bbox = xywh2xyxy(gt_bbox)
            gt_bbox *= 608

            pred_score = pred['scores'].cpu().numpy()
            pred_label = pred['labels'].cpu().numpy()
            pred_bbox = pred['boxes'].cpu().numpy()

            correct = process_batch(np.hstack((pred_bbox, np.expand_dims(pred_score, 1), np.expand_dims(pred_label, 1))),
                                    np.hstack((np.expand_dims(gt_label, 1), gt_bbox)), self.iouv)
            # print((correct, pred_score, pred_label, gt_label))
            self.stats.append((correct, pred_score, pred_label, gt_label))  # (correct, conf, pcls, tcls)

            self.count += 1

    def evaluate(self):
        if self.count < 1:
            return None
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        print(mp, mr, map50, map)
        performances = {}
        performances['precision'] = p.mean()
        performances['recall'] = r.mean()
        performances['mAP50'] = ap50.mean()
        performances['mAP'] = ap.mean()
        performances['performance'] = performances['mAP']
        return performances

    def reset(self):
        self.stats = []
        self.count = 0

if __name__ == '__main__':
    evaluator = Yolov5Evaluator()

    gt_targets_list = []
    pred_list = []
    for si in range(548):
        tbox = torch.load('/home/lmin/pythonCode/scripts/weights/tphyolov5/tbox_' + str(si) + '.pth')
        predn = torch.load('/home/lmin/pythonCode/scripts/weights/tphyolov5/predn_' + str(si) + '.pth')
        labelsn = torch.load('/home/lmin/pythonCode/scripts/weights/tphyolov5/labelsn_' + str(si) + '.pth')

        gt_targets = {}
        gt_targets['boxes'] = tbox
        # gt_targets['boxes'] = labelsn[:, 1:]
        gt_targets['labels'] = labelsn[:, 0]

        pred = {}
        pred['scores'] = predn[:, 4]
        pred['labels'] = predn[:, 5]
        pred['boxes'] = predn[:, :4]

        gt_targets_list.append(gt_targets)
        pred_list.append(pred)


    evaluator.update(gt_targets_list, pred_list)
    evaluator.evaluate()
    print()
