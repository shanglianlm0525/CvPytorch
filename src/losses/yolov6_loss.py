# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/14 15:58
# @Author : liumin
# @File : yolov6_loss.py
import math

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from src.models.anchors.yolov6_anchor_generator import generate_anchors
from src.models.assigners.atss_assigner import ATSSAssigner
from src.models.assigners.tal_assigner import TaskAlignedAssigner


def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


class YOLOv6Loss:
    '''Loss computation func.'''

    def __init__(self, fpn_strides=[8, 16, 32], grid_cell_size=5.0, grid_cell_offset=0.5,num_classes=80,
                 ori_img_size=640, warmup_epoch=4, use_dfl=True, reg_max=16, iou_type='giou',
                 loss_weight={'class': 1.0, 'iou': 2.5, 'dfl': 0.5, 'cwd': 10.0},
                 distill_feat=False, distill_weight={'class': 1.0, 'dfl': 1.0 }):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight
        self.use_extra = False
        self.distill_feat = distill_feat
        self.distill_weight = distill_weight

    def __call__(self,outputs,targets,epoch_num,t_outputs=None,s_featmaps=None,t_featmaps=None,max_epoch=None, temperature=None):
        if t_outputs is not None and s_featmaps is not None and t_featmaps is not None and max_epoch is not None and temperature is not None:
            self.use_extra = True
        feats, pred_scores, pred_distri = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                             device=feats[0].device)
        if self.use_extra:
            t_feats, t_pred_scores, t_pred_distri = t_outputs
            t_anchors, t_anchor_points, t_n_anchors_list, t_stride_tensor = generate_anchors(t_feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset,
                                 device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # targets
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:]  # xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        if self.use_extra:
            t_anchor_points_s = t_anchor_points / t_stride_tensor
            t_pred_bboxes = self.bbox_decode(t_anchor_points_s, t_pred_distri)  # xyxy

        if epoch_num < self.warmup_epoch:
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.warmup_assigner(
                    anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    mask_gt,
                    pred_bboxes.detach() * stride_tensor)
        else:
            target_labels, target_bboxes, target_scores, fg_mask = \
                self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    mask_gt)

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels, self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        loss_cls /= target_scores_sum

        if self.use_extra:
            # bbox loss
            loss_iou, loss_dfl, d_loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes, target_scores, target_scores_sum, fg_mask,
                                                            t_pred_distri, t_pred_bboxes, temperature)

            logits_student = pred_scores
            logits_teacher = t_pred_scores
            distill_num_classes = self.num_classes
            d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher, distill_num_classes, temperature)
            if self.distill_feat:
                d_loss_cw = self.distill_loss_cw(s_featmaps, t_featmaps)
            else:
                d_loss_cw = torch.tensor(0.).to(feats[0].device)
            import math
            distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01 - 1) + 1
            d_loss_dfl *= distill_weightdecay
            d_loss_cls *= distill_weightdecay
            d_loss_cw *= distill_weightdecay
            loss_cls_all = loss_cls + d_loss_cls * self.distill_weight['class']
            loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight['dfl']
            loss = self.loss_weight['class'] * loss_cls_all + \
                   self.loss_weight['iou'] * loss_iou + \
                   self.loss_weight['dfl'] * loss_dfl_all + \
                   self.loss_weight['cwd'] * d_loss_cw

            return loss, \
                   torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                              (self.loss_weight['dfl'] * loss_dfl_all).unsqueeze(0),
                              (self.loss_weight['class'] * loss_cls_all).unsqueeze(0),
                              (self.loss_weight['cwd'] * d_loss_cw).unsqueeze(0))).detach()
        else:
            # bbox loss
            loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                                target_scores, target_scores_sum, fg_mask)

            loss = self.loss_weight['class'] * loss_cls + \
                   self.loss_weight['iou'] * loss_iou + \
                   self.loss_weight['dfl'] * loss_dfl

            return loss, \
                   torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                              (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                              (self.loss_weight['class'] * loss_cls).unsqueeze(0))).detach()

    def distill_loss_cls(self, logits_student, logits_teacher, num_classes, temperature=20):
        logits_student = logits_student.view(-1, num_classes)
        logits_teacher = logits_teacher.view(-1, num_classes)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        d_loss_cls *= temperature ** 2
        return d_loss_cls

    def distill_loss_cw(self, s_feats, t_feats, temperature=1):
        N, C, H, W = s_feats[0].shape
        # print(N,C,H,W)
        loss_cw = F.kl_div(F.log_softmax(s_feats[0].view(N, C, H * W) / temperature, dim=2),
                           F.log_softmax(t_feats[0].view(N, C, H * W).detach() / temperature, dim=2),
                           reduction='sum',
                           log_target=True) * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[1].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[1].view(N, C, H * W) / temperature, dim=2),
                            F.log_softmax(t_feats[1].view(N, C, H * W).detach() / temperature, dim=2),
                            reduction='sum',
                            log_target=True) * (temperature * temperature) / (N * C)

        N, C, H, W = s_feats[2].shape
        # print(N,C,H,W)
        loss_cw += F.kl_div(F.log_softmax(s_feats[2].view(N, C, H * W) / temperature, dim=2),
                            F.log_softmax(t_feats[2].view(N, C, H * W).detach() / temperature, dim=2),
                            reduction='sum',
                            log_target=True) * (temperature * temperature) / (N * C)
        # print(loss_cw)
        return loss_cw

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
            np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
            targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class IOUloss:
    """ Calculate IoU loss.
    """

    def __init__(self, box_format='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            box_format: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid divide by zero error.
        """
        self.box_format = box_format
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [M, 4] and [Nm 4].
        """
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
            elif self.box_format == 'xywh':
                b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
                b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
                b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
                b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        else:
            if self.box_format == 'xyxy':
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)

            elif self.box_format == 'xywh':
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w / 2, b1_x1 + b1_w / 2
                b1_y1, b1_y2 = b1_y1 - b1_h / 2, b1_y1 + b1_h / 2
                b2_x1, b2_x2 = b2_x1 - b2_w / 2, b2_x1 + b2_w / 2
                b2_y1, b2_y2 = b2_y1 - b2_h / 2, b2_y1 + b2_h / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.use_extra = False

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                t_pred_dist=None, t_pred_bboxes=None, temperature=None):
        if t_pred_dist is not None and t_pred_bboxes is not None and temperature is not None:
            self.use_extra = True
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            if self.use_extra:
                t_pred_bboxes_pos = torch.masked_select(t_pred_bboxes, bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / target_scores_sum

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                if self.use_extra:
                    t_pred_dist_pos = torch.masked_select(t_pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                         target_ltrb_pos) * bbox_weight
                loss_dfl = loss_dfl.sum() / target_scores_sum
                if self.use_extra:
                    d_loss_dfl = self.distill_loss_dfl(pred_dist_pos, t_pred_dist_pos, temperature) * bbox_weight
                    d_loss_dfl = d_loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = torch.tensor(0.).to(pred_dist.device)
                if self.use_extra:
                    d_loss_dfl = torch.tensor(0.).to(pred_dist.device)

        else:

            loss_iou = torch.tensor(0.).to(pred_dist.device)
            loss_dfl = torch.tensor(0.).to(pred_dist.device)
            if self.use_extra:
                d_loss_dfl = torch.tensor(0.).to(pred_dist.device)

        if self.use_extra:
            return loss_iou, loss_dfl, d_loss_dfl
        else:
            return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def distill_loss_dfl(self, logits_student, logits_teacher, temperature=20):

        logits_student = logits_student.view(-1, 17)
        logits_teacher = logits_teacher.view(-1, 17)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature ** 2
        return d_loss_dfl