# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/8 20:12
# @Author : liumin
# @File : ssd_detect.py

import torch
import torch.nn.functional as F
import torchvision

from src.data.transforms.det_target_transforms import center_form_to_corner_form


def convert_locations_to_boxes(locations, priors, center_variance,
                               size_variance):
    """Convert regressional location results of SSD into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output of SSD. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)

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


class PostProcessor:
    def __init__(self, top_k=100, nms_thresh=0.01, conf_thresh=0.45, variance = [0.1, 0.2]):
        super().__init__()
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.variance = variance

    def __call__(self, cls_logits, box_logits, priors):
        batch_scores = F.softmax(cls_logits, dim=2)
        batch_boxes = convert_locations_to_boxes(box_logits, priors, self.variance[0],self.variance[1])
        batch_boxes = center_form_to_corner_form(batch_boxes)

        # batches_scores, batches_boxes = detections
        device = batch_scores.device
        batch_size = batch_scores.size(0)
        results = []
        for batch_id in range(batch_size):
            scores, boxes = batch_scores[batch_id], batch_boxes[batch_id]  # (N, #CLS) (N, 4)
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

            keep = batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.top_k]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = dict(boxes=boxes, labels=labels, scores=scores)
            results.append(container)
        return results