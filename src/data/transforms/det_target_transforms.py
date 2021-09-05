# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/7 16:52
# @Author : liumin
# @File : det_target_transforms.py

import torch
import numpy as np

from src.models.anchors.prior_box import PriorBox


__all__ = ['Compose', 'SSDTargetTransform']


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample



def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: num_priors x num_targets

    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


class SSDTargetTransform:
    def __init__(self, image_size=300, feature_maps=[38, 19, 10, 5, 3, 1],
                 min_sizes=[21, 45, 99, 153, 207, 261], max_sizes=[45, 99, 153, 207, 261, 315],
                 strides=[8, 16, 32, 64, 100, 300], aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], clip=True,
                 center_variance=0.1, size_variance=0.2, iou_threshold=0.5):
        self.center_form_priors = PriorBox(image_size=image_size, feature_maps=feature_maps,
                 min_sizes=min_sizes, max_sizes=max_sizes,
                 strides=strides, aspect_ratios=aspect_ratios, clip=clip)()
        self.corner_form_priors = center_form_to_corner_form(self.center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        classes = target["labels"]
        if type(boxes) is np.ndarray:
            boxes = torch.from_numpy(boxes)
        if type(classes) is np.ndarray:
            classes = torch.from_numpy(classes)
        boxes_pri, classes_pri = assign_priors(boxes, classes, self.corner_form_priors, self.iou_threshold)
        boxes_pri = corner_form_to_center_form(boxes_pri)
        boxes_pri = convert_boxes_to_locations(boxes_pri, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        target["boxes"] = boxes_pri
        target["labels"] = classes_pri
        target["anchors"] = self.center_form_priors
        return {'image': img, 'target': target}