# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/28 9:40
# @Author : liumin
# @File : objectbox_loss.py

import torch
import torch.nn as nn
import numpy as np


class ObjectBoxLoss(nn.Module):
    # Compute losses
    def __init__(self, num_classes, stride=[8., 16., 32.], num_layers=3, num_anchors=1, device='cuda:0', autobalance=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_anchors = num_anchors  # number of anchors
        self.stride = stride
        self.device = device

        self.hyp_cls_pw, self.hyp_obj_pw, self.hyp_label_smoothing, self.hyp_fl_gamma = 1.0, 1.0, 0.0, 0.0

        self.hyp_anchor_t = 4.0
        self.hyp_box, self.hyp_obj, self.hyp_cls = 0.05, 1.0, 1.0

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp_cls_pw], device=self.device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp_obj_pw], device=self.device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=self.hyp_label_smoothing)  # positive, negative BCE targets

        # Focal loss
        g = self.hyp_fl_gamma  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(self.num_layers, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
        self.ssi = list(self.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = BCEcls, BCEobj, 1.0, autobalance

    def forward(self, p, targets):  # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

        tcls_center, indices_center, tbox_center = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions

            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            b_1, a_1, gj_1, gi_1 = indices_center[i]  # image, 0, gridy, gridx
            n = b_1.shape[0]  # number of targets

            if n:
                ps_center = pi[b_1, a_1, gj_1, gi_1]  # prediction subset corresponding to targets
                # Regression
                s_gain = torch.tensor([2 ** i, 2 ** i], device=pi.device).repeat((n, 2))
                pbox = (ps_center[:, :4].sigmoid() * 2) ** 2 * s_gain
                iou_center = sd_iou(pbox.T, tbox_center[i], x1y1x2y2=True, PIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou_center).mean()  # iou loss

                # Objectness
                score_iou_center = iou_center.detach().clamp(0).type(tobj.dtype)

                tobj[b_1, a_1, gj_1, gi_1] = (1.0 - self.gr) + self.gr * score_iou_center  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps_center[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls_center[i]] = self.cp
                    lcls += self.BCEcls(ps_center[:, 5:], t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj los

            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp_box
        lobj *= self.hyp_obj
        lcls *= self.hyp_cls
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        num_nobox = 1  # used in ablation

        nt = targets.shape[0]  #
        tcls_center, indices_center, tbox_center = [], [], []

        gain_i = torch.ones(6, device=targets.device)  # normalized to gridspace gain

        gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
        ai = torch.arange(num_nobox, device=targets.device).float().view(num_nobox, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets_b = torch.cat((targets.repeat(num_nobox, 1, 1), ai[:, :, None]), 2)  #

        g = 0.5
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                            ], device=targets.device).float() * g  # offsets

        for i in range(self.num_layers):
            nB = p[i].shape[0]
            nG0 = p[i].shape[2]
            nG1 = p[i].shape[3]

            gain_i[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            targets_i = targets * gain_i
            t_boxes = targets_i[:, 2:6]

            # xywh >> x1y1x2y2
            t_boxes_xyxy = xywh2xyxy(t_boxes)
            xmin, ymin, xmax, ymax = t_boxes_xyxy[:, 0], t_boxes_xyxy[:, 1], t_boxes_xyxy[:, 2], t_boxes_xyxy[:, 3]

            # used in ablation
            xmin_grid = torch.clamp(xmin.long(), min=0, max=nG1 - 1)
            ymin_grid = torch.clamp(ymin.long(), min=0, max=nG0 - 1)
            xmax_grid = torch.clamp(xmax.long(), min=0, max=nG1 - 1)
            ymax_grid = torch.clamp(ymax.long(), min=0, max=nG0 - 1)

            allwidth = (ymax_grid - ymin_grid) + 1

            th1 = 0  # used in ablation
            obj_i = (allwidth >= th1)

            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
            t = targets_b * gain

            xyxy = (torch.stack((xmin, ymin, xmax, ymax)).T)
            xyxy = xyxy[None, :].repeat((1, 1, 1))
            t = torch.cat((t, xyxy), 2)

            mask = torch.zeros(t.shape[1], dtype=torch.bool)
            mask[obj_i] = True
            t = t[:, mask]

            t1 = t.view(t.shape[0] * t.shape[1], t.shape[2])

            # Offsets
            gxy = t1[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m, j & k, j & m, l & k, l & m))
            t1 = t1.repeat((9, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]

            ####
            gxy = t1[:, 2:4]  # grid xy
            gij = (gxy - offsets).long()

            gxy_a = (gxy - offsets)
            j, k = ((gxy_a % 1. < g) & (gxy_a > 1.)).T

            jk = torch.stack((j, k)).T
            mask = 2 ** torch.arange(1, -1, -1).to(jk.device, jk.dtype)
            a = torch.sum(mask * jk, -1)

            t2 = torch.clone(t1)
            t2[:, 6] = a + 1
            t1 = torch.cat((t1, t2))
            ####

            # Define
            b, c = t1[:, :2].long().T  # image, class
            gxy = t1[:, 2:4]  # grid xy
            offsets = torch.cat((offsets, offsets))
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            gij_xmin = t1[:, 7]
            gij_ymin = t1[:, 8]
            gij_xmax = t1[:, 9]
            gij_ymax = t1[:, 10]

            dx1 = (((gij[:, 0] + 1) - gij_xmin))  # / nG1)
            dy1 = (((gij[:, 1] + 1) - gij_ymin))  # / nG0)
            dx2 = ((gij_xmax - (gij[:, 0])))  # / nG1)
            dy2 = ((gij_ymax - (gij[:, 1])))  # / nG0)

            # Append
            l = 0
            indices_center.append((b, l, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox_center.append(torch.stack((dx1, dy1, dx2, dy2)).T)  # box
            tcls_center.append(c)  # class

        return tcls_center, indices_center, tbox_center


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def sd_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, PIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    S = ((b2_x1 - b1_x1) ** 2) + ((b2_y1 - b1_y1) ** 2) + ((b2_x2 - b1_x2) ** 2) + ((b2_y2 - b1_y2) ** 2)
    inter_x1, inter_y1 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1)
    inter_x2, inter_y2 = torch.min(b1_x2, b2_x2), torch.min(b1_y2, b2_y2)
    I = (inter_x1 + inter_x2 - 1) ** 2 + (inter_y1 + inter_y2 - 1) ** 2

    cw = torch.max(b1_x1, b2_x1) + torch.max(b1_x2, b2_x2) - 1  # convex (smallest enclosing box) width
    ch = torch.max(b1_y1, b2_y1) + torch.max(b1_y2, b2_y2) - 1  # convex height
    C = cw ** 2 + ch ** 2 + eps

    rho = 1  # used in development, can be removed
    iou = (I - (rho * S)) / C

    return iou
