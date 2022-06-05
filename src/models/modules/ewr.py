# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/22 17:28
# @Author : liumin
# @File : ewr.py

import torch
import torch.nn as nn
from torch import distributed as dist
import torch.nn.functional as F


def get_loss(head_outs, gt_meta):
    pred_scores, pred_distri, anchors, \
    anchor_points, num_anchors_list, stride_tensor = head_outs
    device = pred_scores.device
    anchors = anchors.to(device)
    anchor_points = anchor_points.to(device)
    stride_tensor = stride_tensor.to(device)

    anchor_points_s = anchor_points / stride_tensor
    pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

    gt_labels = gt_meta['gt_class']
    gt_labels = gt_labels.to(torch.int64)
    gt_bboxes = gt_meta['gt_bbox']
    pad_gt_mask = gt_meta['pad_gt_mask']

    # miemie2013: 剪掉填充的gt
    num_boxes = pad_gt_mask.sum([1, 2])
    num_max_boxes = num_boxes.max().to(torch.int32)
    pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
    gt_labels = gt_labels[:, :num_max_boxes, :]
    gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

    # label assignment
    if gt_meta['epoch_id'] < self.static_assigner_epoch:
        assigned_labels, assigned_bboxes, assigned_scores = \
            self.static_assigner(
                anchors,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes,
                pred_bboxes=pred_bboxes.detach() * stride_tensor)
        alpha_l = 0.25
        # import numpy as np
        # dic = np.load('../aa2.npz')
        # print_diff(dic, 'assigned_labels', assigned_labels)
        # print_diff(dic, 'assigned_bboxes', assigned_bboxes)
        # print_diff(dic, 'assigned_scores', assigned_scores)
        # assigned_labels = torch.Tensor(dic['assigned_labels']).to(torch.int64)
        # assigned_bboxes = torch.Tensor(dic['assigned_bboxes']).to(torch.float32)
        # assigned_scores = torch.Tensor(dic['assigned_scores']).to(torch.float32)
        # print()
    else:
        assigned_labels, assigned_bboxes, assigned_scores = \
            self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
        alpha_l = -1
    # rescale bbox
    assigned_bboxes /= stride_tensor
    # cls loss
    if self.use_varifocal_loss:
        one_hot_label = F.one_hot(assigned_labels,
                                  self.num_classes + 1)[..., :-1]
        loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                        one_hot_label)
    else:
        loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

    # 每张卡上的assigned_scores_sum求平均，而且max(x, 1)
    assigned_scores_sum = assigned_scores.sum()
    world_size = dist.get_world_size()
    if world_size > 1:
        dist.all_reduce(assigned_scores_sum, op=dist.ReduceOp.SUM)
        assigned_scores_sum = assigned_scores_sum / world_size
    assigned_scores_sum = F.relu(assigned_scores_sum - 1.) + 1.  # y = max(x, 1)
    loss_cls /= assigned_scores_sum

    loss_l1, loss_iou, loss_dfl = \
        self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                        assigned_labels, assigned_bboxes, assigned_scores,
                        assigned_scores_sum)

    loss = self.loss_weight['class'] * loss_cls + \
           self.loss_weight['iou'] * loss_iou + \
           self.loss_weight['dfl'] * loss_dfl
    out_dict = {
        'total_loss': loss,
        'loss_cls': loss_cls,
        'loss_iou': loss_iou,
        'loss_dfl': loss_dfl,
        'loss_l1': loss_l1,
    }
    return out_dict