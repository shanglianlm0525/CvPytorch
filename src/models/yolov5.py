# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/8 16:15
# @Author : liumin
# @File : yolov5.py
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import math
from CvPytorch.src.models.detects.yolov5_detect import Detect
from CvPytorch.src.models.yolov5_model import parse_model, check_anchor_order, initialize_weights


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-12):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2] + eps, box1[3] + eps
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2] + eps, box2[3] + eps
    else:  # nanodet_transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2 + eps
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2 + eps
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2 + eps
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def wh_iou(box1, box2):
    # Returns the IoU of wh1 to wh2. wh1 is 2, wh2 is nx2
    box2 = box2.t()

    # w, h = box1
    w1, h1 = box1[0], box1[1]
    w2, h2 = box2[0], box2[1]

    # Intersection area
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)

    # Union Area
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area

    return inter_area / union_area  # iou


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn() https://arxiv.org/pdf/1708.02002.pdf
    # i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=2.5)
    def __init__(self, loss_fcn, gamma=0.5, alpha=1, reduction='mean'):
        super(FocalLoss, self).__init__()
        loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = self.loss_fcn(input, target)
        loss *= self.alpha * (1.000001 - torch.exp(-loss)) ** self.gamma  # non-zero power for gradient stability

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def build_targets(p, targets, model):
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    det = model.module.model[-1] if is_parallel(model) else model[-1]  # Detect() module
    na, nt = det.na, targets.shape[0]  # number of anchors, targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices

    g = 0.5  # bias
    off = torch.tensor([[0, 0],
                        [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                        # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                        ], device=targets.device).float() * g  # offsets

    for i in range(det.nl):
        anchors = det.anchors[i]
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

        # Match targets to anchors
        t = targets * gain
        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = torch.max(r, 1. / r).max(2)[0] < 2.91 # model.hyp['anchor_t']  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = torch.stack((torch.ones_like(j), j, k, l, m))
            t = t.repeat((5, 1, 1))[j]
            offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = targets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].long().T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).long()
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].long()  # anchor indices
        indices.append((b, a, gj, gi))  # image, anchor, grid indices
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        anch.append(anchors[a])  # anchors
        tcls.append(c)  # class

    return tcls, tbox, indices, anch


def compute_loss(p, targets, model, num_classes):  # predictions, targets, model
    device = targets.device
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    tcls, tbox, indices, anchors = build_targets(p, targets, model)  # targets
    # hyperparameters
    h_cls_pw = 0.631
    h_obj_pw = 0.911
    h_fl_gamma = 0.0
    h_giou = 0.0296
    h_obj = 0.301
    h_cls = 0.243
    gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)

    # Define criteria
    BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h_cls_pw])).to(device)
    BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([h_obj_pw])).to(device)

    # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
    cp, cn = smooth_BCE(eps=0.0)

    # Focal loss
    g = h_fl_gamma  # focal loss gamma
    if g > 0:
        BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

    # Losses
    nt = 0  # number of targets
    np = len(p)  # number of outputs
    balance = [4.0, 1.0, 0.4] if np == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6
    for i, pi in enumerate(p):  # layer index, layer predictions
        b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
        tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

        n = b.shape[0]  # number of targets
        if n:
            nt += n  # cumulative targets
            ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

            # Regression
            pxy = ps[:, :2].sigmoid() * 2. - 0.5
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
            pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
            giou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # giou(prediction, target)
            lbox += (1.0 - giou).mean()  # giou loss

            # Objectness
            tobj[b, a, gj, gi] = (1.0 - gr) + gr * giou.detach().clamp(0).type(tobj.dtype)  # giou ratio

            # Classification
            if num_classes > 1:  # cls loss (only if multiple classes)
                t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                t[range(n), tcls[i]] = cp
                lcls += BCEcls(ps[:, 5:], t)  # BCE

            # Append targets to text file
            # with open('targets.txt', 'a') as file:
            #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

        lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

    s = 3 / np  # output count scaling
    lbox *= h_giou * s
    lobj *= h_obj * s * (1.4 if np == 4 else 1.)
    lcls *= h_cls * s
    bs = tobj.shape[0]  # batch size

    loss = lbox + lobj + lcls
    return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()



class YOLOV5(nn.Module):
    def __init__(self, dictionary=None):
        super(YOLOV5, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        cfg = 'src/models/yolov5s.yaml'
        ch = 3

        import yaml  # for torch hub
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.FullLoader)  # model dict

        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 128  # 2x min stride
            m.stride = torch.tensor([8, 16, 32], dtype=torch.float)  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if mode == 'infer':

            return
        else:
            losses = {}

            x = imgs
            y = []  # outputs
            for i, m in enumerate(self.model):
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                y.append(x if m.i in self.save else None)  # save output

            bboxes = targets['gt_bboxes']
            for bbox, img in zip(bboxes, imgs):
                bbox[:, :] = xyxy2xywh(bbox[:, :])  # convert xyxy to xywh
                bbox[:, [0, 2]] /= img.shape[2]  # normalized width 0-1
                bbox[:, [1, 3]] /= img.shape[1]  # normalized height 0-1

            gts = None
            for i, (lbl, bbox) in enumerate(zip(targets['gt_labels'], bboxes)):
                gt = torch.cat([torch.as_tensor(i).to(lbl.device).repeat(lbl.shape[0]).unsqueeze(1),
                               lbl.unsqueeze(1), bbox], dim=1)
                if i == 0:
                    gts = gt
                else:
                    gts = torch.cat([gts, gt], dim=0)

            loss, loss_items = compute_loss(x, gts, self.model, self._num_classes)  # scaled by batch_size

            losses['bbox_loss'] = loss_items[0]
            losses['obj_loss'] = loss_items[1]
            losses['cls_loss'] = loss_items[2]
            losses['loss'] = loss

            if mode == 'val':
                return losses
            else:

                return losses