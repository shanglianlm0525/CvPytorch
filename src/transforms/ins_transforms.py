# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/11 14:36
# @Author : liumin
# @File : ins_transforms.py

import random

import cv2
import torch
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image


__all__ = ['RandomHorizontalFlip',
        'ColorJitter', 'RandomGaussianBlur',
        'Normalize', 'DeNormalize', 'ToTensor']

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img, target = sample['image'], sample['target']
        img = F.to_tensor(img.astype(np.uint8))
        return {'image': img, 'target': target}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img,target = sample['image'], sample['target']
        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': img,'target': target}


class DeNormalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        for t, m, s in zip(img, self.mean, self.std):
            t.mul_(s).add_(m)
        return {'image': img,'target': target}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target



def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            h, w, _ = img.shape
            img = cv2.flip(img, 1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = w - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, w)
                target["keypoints"] = keypoints
        return {'image': img,'target': target}


class RandomGaussianBlur(object):
    def __init__(self, p=0.5, radius=5):
        self.p = p
        self.radius = radius

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.GaussianBlur(img, (self.radius, self.radius), 0)
        return {'image': img,'target': target}


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = Image.fromarray(img)
        img = T.ColorJitter(self.brightness, self.contrast, self.saturation,self.hue)(img)
        img = np.asarray(img, dtype=np.int32)
        return {'image': img, 'target': target}

class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return {'image': img,'target': target}


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return {'image': img,'target': target}

