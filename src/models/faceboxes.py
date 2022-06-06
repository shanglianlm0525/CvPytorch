# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/4 15:12
# @Author : liumin
# @File : faceboxes.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import product
from math import ceil

import torchvision

from src.models.modules.convs import ConvModule


"""
    FaceBoxes: A CPU Real-time Face Detector with High Accuracy
    https://arxiv.org/pdf/1708.05234.pdf
"""

# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances=[0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat((priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
        priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])), loc.dim() - 1)
    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]
    return boxes


class PriorBox(object):
    def __init__(self, min_sizes=[[32, 64, 128], [256], [512]], steps=[32, 64, 128], clip = False, image_size = [1024, 1024]):
        super(PriorBox, self).__init__()
        self.min_sizes = min_sizes
        self.steps = steps
        self.clip = clip
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        loc_t = loc_t.cuda()
        conf_t = conf_t.cuda()

        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c


class Inception(nn.Module):
    def __init__(self, in_channels=128, mid_channels= 24):
        super(Inception, self).__init__()
        out_channels = in_channels // 4
        self.branch1 = ConvModule(in_channels, out_channels, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU')

        self.branch2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvModule(in_channels, out_channels, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU')
        )

        self.branch3 = nn.Sequential(
            ConvModule(in_channels, mid_channels, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU'),
            ConvModule(mid_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')
        )

        self.branch4 = nn.Sequential(
            ConvModule(in_channels, mid_channels, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU'),
            ConvModule(mid_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU'),
            ConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='ReLU')
        )

    def forward(self, x):
       x1 = self.branch1(x)
       x2 = self.branch2(x)
       x3 = self.branch3(x)
       x4 = self.branch4(x)
       return torch.cat([x1, x2, x3, x4], dim=1)


class Multibox(nn.Module):
    def __init__(self, num_classes=2):
        super(Multibox, self).__init__()
        self.num_classes = num_classes

        self.loc_layers = nn.Sequential(
            nn.Conv2d(128, 21 * 4, 3, 1, 1),
            nn.Conv2d(256, 1 * 4, 3, 1, 1),
            nn.Conv2d(256, 1 * 4, 3, 1, 1)
        )
        self.conf_layers = nn.Sequential(
            nn.Conv2d(128, 21 * num_classes, 3, 1, 1),
            nn.Conv2d(256, 1 * num_classes, 3, 1, 1),
            nn.Conv2d(256, 1 * num_classes, 3, 1, 1)
        )

    def forward(self, x):
        loc = list()
        conf = list()
        for (xx, l, c) in zip(x, self.loc_layers, self.conf_layers):
            loc.append(l(xx).permute(0, 2, 3, 1).contiguous())
            conf.append(c(xx).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # test: output = (loc.view(loc.size(0), -1, 4), self.softmax(conf.view(conf.size(0), -1, self.num_classes)))
        # train: output = (loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes))
        boxes = loc.view(loc.size(0), -1, 4)
        scores = conf.view(conf.size(0), -1, self.num_classes)
        return boxes, scores


def nms(boxes, scores, nms_thresh):
    """ Performs non-maximum suppression, run on GPU or CPU according to
    boxes's device.
    Args:
        boxes(Tensor[N, 4]): boxes in (x1, y1, x2, y2) format, use absolute coordinates(or relative coordinates)
        scores(Tensor[N]): scores
        nms_thresh(float): thresh
    Returns:
        indices kept.
    """
    keep = torchvision.ops.nms(boxes, scores, nms_thresh)
    return keep


def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep

class FaceBoxesDetect(nn.Module):
    def __init__(self, num_classes, top_k=750, iou_thresh=0.01, conf_thresh=0.3, filter_thresh = 0.05, variance = [0.1, 0.2]):
        super(FaceBoxesDetect, self).__init__()
        self.num_classes = num_classes
        self.top_k = top_k
        # Parameters used in nms.
        self.filter_thresh = filter_thresh
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance

    def forward(self, boxes_logits, cls_logits, priors):
        scores_logits = F.softmax(cls_logits, dim=2)
        boxes_logits = decode(boxes_logits, priors)
        boxes_logits[..., 0::2] *= 1024
        boxes_logits[..., 1::2] *= 1024

        device = scores_logits.device
        batch_size = scores_logits.size(0)

        pred_boxes, pred_scores, pred_labels = [], [], []
        for bid in range(batch_size):
            scores, boxes = scores_logits[bid], boxes_logits[bid]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            indices = torch.nonzero(scores > self.conf_thresh).squeeze(1)
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            keep = batched_nms(boxes, scores, labels, self.iou_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.top_k]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # print(boxes, scores, labels)
            pred_boxes.append(boxes)
            pred_scores.append(scores)
            pred_labels.append(labels)

        return pred_boxes, pred_scores, pred_labels





class FaceBoxes(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(FaceBoxes, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 1024, 1024)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        '''
        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)
        self.detect = build_detect(self.model_cfg.DETECT)
        self.loss = build_loss(self.model_cfg.LOSS)
        '''

        # Rapidly Digested Convolutional Layers
        self.rdcl = nn.Sequential(
            ConvModule(3, 24, 7, 4, 3, norm_cfg=dict(type='BN'), activation='CReLU'),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvModule(48, 64, 5, 2, 2, norm_cfg=dict(type='BN'), activation='CReLU'),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Multiple Scale Convolutional Layers
        self.mscl1 = nn.Sequential(
            Inception(),
            Inception(),
            Inception()
        )

        self.mscl2 = nn.Sequential(ConvModule(128, 128, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU'),
                                   ConvModule(128, 256, 3, 2, 1, norm_cfg=dict(type='BN'), activation='ReLU'))

        self.mscl3 = nn.Sequential(ConvModule(256, 128, 1, 1, 0, norm_cfg=dict(type='BN'), activation='ReLU'),
                                   ConvModule(128, 256, 3, 2, 1, norm_cfg=dict(type='BN'), activation='ReLU'))

        self.head = Multibox(self.num_classes)

        self.priors= PriorBox().forward().cuda()
        self.criterion = MultiBoxLoss(self.num_classes, 0.35, True, 0, True, 7, 0.35, False)

        self.detect = FaceBoxesDetect(self.num_classes)
        self.conf_thres = 0.05  # confidence threshold
        self.iou_thres = 0.6  # NMS IoU threshold
        self.topK = 5000
        self.keep_topK = 750
        self.init_weights()

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.xavier_normal_(m.weight.data)
                    m.bias.data.fill_(0.02)
                else:
                    m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def trans_specific_format(self, imgs, targets):
        new_gts = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for i, target in enumerate(targets):
            new_gt = torch.zeros((target['labels'].shape[0], 5), device=target['labels'].device)
            new_gt[:, :] = torch.cat([target['boxes'], target['labels'].unsqueeze(1)], 1)
            new_gts.append(new_gt)
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["gts"] = new_gts
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return torch.stack(imgs), t_targets


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.5
        if mode == 'infer':
            pass
        else:
            losses = {}
            imgs, targets = self.trans_specific_format(imgs, targets)

            feature = self.rdcl(imgs)
            C1 = self.mscl1(feature)
            C2 = self.mscl2(C1)
            C3 = self.mscl3(C2)
            boxes_logits, cls_logits = self.head([C1, C2, C3])

            if mode == 'train':
                losses['reg_loss'], losses['cls_loss'] = self.criterion((boxes_logits, cls_logits), self.priors, targets['gts'])
                losses['loss'] = 2.0 * losses['reg_loss'] + losses['cls_loss']
            else:
                losses['loss'] = torch.tensor(0).cuda()
            # print(losses)

            if mode == 'val':
                # print('-'*30)
                # print('targets', targets['gts'])
                # print('boxes_logits, cls_logits', boxes_logits, cls_logits)
                pred_boxes, pred_scores, pred_labels = self.detect(boxes_logits, cls_logits, self.priors)
                # print('pred_boxes, pred_scores, pred_labels', pred_boxes, pred_scores, pred_labels)

                outputs = []
                for i, (width, height, scale, pad, pred_box, pred_label, pred_score) in enumerate(
                        zip(targets['width'], targets['height'], targets['scales'], targets['pads'],
                            pred_boxes, pred_labels, pred_scores)):
                    scale = scale.cpu().numpy()
                    pad = pad.cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    pred_box = pred_box.clone()

                    # print('pred_box', pred_box)
                    bboxes_np = pred_box[:, :4].cpu().numpy()
                    bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                    bboxes_np[:, [1, 3]] -= pad[0]
                    bboxes_np[:, [0, 2]] /= scale[1]
                    bboxes_np[:, [1, 3]] /= scale[0]

                    # clip boxes
                    bboxes_np[..., 0::2] = bboxes_np[..., 0::2].clip(0, width)
                    bboxes_np[..., 1::3] = bboxes_np[..., 1::3].clip(0, height)

                    # print('bboxes_np', bboxes_np)
                    keep = pred_score > threshold
                    outputs.append({"boxes": torch.tensor(bboxes_np)[keep], "labels": pred_label[keep], "scores": pred_score[keep]})

                return losses, outputs
            else:
                return losses