# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/16 14:18
# @Author : liumin
# @File : eval_detection.py

import numpy as np
import torch


def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # expands dim
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # overlap
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # compute area
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # compute iou
    iou = overlap / (area_a + area_b - overlap)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


class VOCEvaluator(object):
    def __init__(self, dictionary, iou_thread=0.5):
        self.dictionary = dictionary
        self.num_class = len(self.dictionary)
        self.iou_thread = iou_thread
        self.gt_labels = []
        self.gt_bboxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.pred_bboxes = []

    def add_batch(self, gt_targets, preds):
        gt_targets = gt_targets.cpu().numpy()
        gt_bbox, gt_label = np.split(gt_targets,[4],2)
        pred_label, pred_score, pred_bbox = preds[0].cpu().numpy(),preds[1].cpu().numpy(),preds[2].cpu().numpy()
        self.gt_labels.append(gt_label)
        self.gt_bboxes.append(gt_bbox)
        self.pred_labels.append(pred_label)
        self.pred_scores.append(pred_score)
        self.pred_bboxes.append(pred_bbox)

    def Precision(self):
        """
            :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
            :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
            :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
            :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
            :param pred_scores: list of 1d array,shape[(m),(n)...]
            :param iou_thread: eg. 0.5
            :param num_cls: eg. 4, total number of class including background which is equal to 0
            :return: a dict containing average precision for each cls
            """
        self.all_ap = {}
        for idx, d in enumerate(self.dictionary[1:]):
            for _label, _weight in d.items():
                # get samples with specific label
                true_label_loc = [sample_labels == idx for sample_labels in self.gt_labels]
                gt_single_cls = [sample_boxes[mask.squeeze(2), :] for sample_boxes, mask in
                                 zip(self.gt_bboxes, true_label_loc)]

                pred_label_loc = [sample_labels == idx for sample_labels in self.pred_labels]
                bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(self.pred_bboxes, pred_label_loc)]
                scores_single_cls = [sample_scores[mask] for sample_scores, mask in
                                     zip(self.pred_scores, pred_label_loc)]

                fp = np.zeros((0,))
                tp = np.zeros((0,))
                scores = np.zeros((0,))
                total_gts = 0
                # loop for each sample
                for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls,
                                                                      scores_single_cls):
                    total_gts = total_gts + len(sample_gts)
                    assigned_gt = []  # one gt can only be assigned to one predicted bbox
                    # loop for each predicted bbox
                    for index in range(len(sample_pred_box)):
                        scores = np.append(scores, sample_scores[index])
                        if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                            continue
                        pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                        iou = iou_2d(sample_gts, pred_box)
                        gt_for_box = np.argmax(iou, axis=0)
                        max_overlap = iou[gt_for_box, 0]
                        if max_overlap >= self.iou_thread and gt_for_box not in assigned_gt:
                            fp = np.append(fp, 0)
                            tp = np.append(tp, 1)
                            assigned_gt.append(gt_for_box)
                        else:
                            fp = np.append(fp, 1)
                            tp = np.append(tp, 0)
                # sort by score
                indices = np.argsort(-scores)
                fp = fp[indices]
                tp = tp[indices]
                # compute cumulative false positives and true positives
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)
                # compute recall and precision
                recall = tp / (total_gts + np.finfo(np.float64).eps)
                precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
                ap = _compute_ap(recall, precision)
                self.all_ap[_label] = ap
                # print(recall, precision)
        return self.all_ap

    def Mean_Precision(self):
        mAP = 0.
        for class_id, class_mAP in self.all_ap.items():
            mAP += float(class_mAP)
        mAP /= (self.num_class - 1)
        return mAP

    def get(self):
        performances = self.Precision()
        performances['mAP'] = self.Mean_Precision()
        performances['performance'] = performances['mAP']
        return performances

    def reset(self):
        self.gt_labels = []
        self.gt_bboxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.pred_bboxes = []


class COCOEvaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.gt_labels = []
        self.gt_bboxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.pred_bboxes = []

    def add_batch(self, gt_label,gt_bbox, pred_label,pred_score,pred_bbox):
        pass

    def Average_Precision(self):
        pass

    def Mean_Average_Precision(self):
        pass

    def reset(self):
        self.gt_labels = []
        self.gt_bboxes = []
        self.pred_labels = []
        self.pred_scores = []
        self.pred_bboxes = []