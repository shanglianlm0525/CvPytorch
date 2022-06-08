# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/23 8:23
# @Author : liumin
# @File : ppyoloe_loss.py

import torch
import torch.nn as nn
from torch import distributed as dist
import torch.nn.functional as F

from src.models.modules.ppyoloe_modules import batch_distance2bbox, GIoULoss


class PPYOLOELoss:
    # Compute losses
    def __init__(self, num_classes=80, static_assigner='ATSSAssigner', assigner='TaskAlignedAssigner',
                 use_varifocal_loss=True, reg_max=16):
        super(PPYOLOELoss, self).__init__()
        self.num_classes = num_classes
        self.static_assigner = static_assigner
        self.assigner = assigner
        self.static_assigner_epoch = 4
        self.use_varifocal_loss = use_varifocal_loss
        self.reg_max = reg_max
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        self.loss_weight = { 'class': 1.0, 'iou': 2.5, 'dfl': 0.5 }
        self.iou_loss = GIoULoss()


    def __call__(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors, \
        anchor_points, num_anchors_list, stride_tensor = head_outs
        device = pred_scores.device
        anchors = anchors.to(device)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['labels']
        gt_labels = gt_labels.to(torch.int64)
        gt_bboxes = gt_meta['boxes']
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
                self.static_assigner(anchors, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask,
                    bg_index=self.num_classes, pred_bboxes=pred_bboxes.detach() * stride_tensor)
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
                self.assigner(pred_scores.detach(), pred_bboxes.detach() * stride_tensor,
                    anchor_points, num_anchors_list, gt_labels, gt_bboxes, pad_gt_mask, bg_index=self.num_classes)
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
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = pred_dist.shape
        device = pred_dist.device
        pred_dist = pred_dist.reshape([b, l, 4, self.reg_max + 1])
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.matmul(self.proj.to(device))
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = F.softmax(pred_dist, dim=-1)
        target_left_onehot = F.one_hot(target_left, pred_dist_act.shape[-1])
        target_right_onehot = F.one_hot(target_right, pred_dist_act.shape[-1])
        loss_left = target_left_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_left = loss_left.sum(-1) * weight_left
        loss_right = loss_right.sum(-1) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([]).to(pred_dist.device)
            loss_iou = torch.zeros([]).to(pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
            # loss_l1 = None
            # loss_iou = None
            # loss_dfl = None
        return loss_l1, loss_iou, loss_dfl


    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        # loss = F.binary_cross_entropy(
        #     score, label, weight=weight, reduction='sum')

        score = score.to(torch.float32)
        eps = 1e-9
        loss = label * (0 - torch.log(score + eps)) + \
               (1.0 - label) * (0 - torch.log(1.0 - score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # loss = F.binary_cross_entropy(
        #     pred_score, gt_score, weight=weight, reduction='sum')

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = pred_score.to(torch.float32)
        eps = 1e-9
        loss = gt_score * (0 - torch.log(pred_score + eps)) + \
               (1.0 - gt_score) * (0 - torch.log(1.0 - pred_score + eps))
        loss *= weight
        loss = loss.sum()
        return loss
