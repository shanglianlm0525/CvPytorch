# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/18 15:35
# @Author : liumin
# @File : keypoint_transforms.py
import copy
import random
import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from pycocotools import mask as coco_mask

__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Normalize', 'ToTensor',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']

def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (array[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple or List[height, width]): size of the image

    Returns:
        array[N, 4]: clipped boxes
    """
    height, width = size
    boxes[..., 0::2] = boxes[..., 0::2].clip(min=0, max=width)
    boxes[..., 1::2] = boxes[..., 1::2].clip(min=0, max=height)
    return boxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        target["boxes"] = torch.from_numpy(target["boxes"])
        if target.__contains__("masks"):
            mask = target["masks"]
            mask = mask.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            mask = np.ascontiguousarray(mask)
            target["masks"] = torch.from_numpy(mask)
        if self.normalize:
            return {'image': torch.from_numpy(img.astype(np.float32)).div_(255.0), 'target': target}
        else:
            return {'image': torch.from_numpy(img.astype(np.float32)), 'target': target }


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        return {'image': F.normalize(img, self.mean, self.std),'target': target}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            height, width, _ = img.shape
            boxes = target["boxes"]
            keypoints = target["keypoints"]
            if boxes.shape[0] != 0:
                xmin = width - 1 - boxes[:, 2]
                xmax = width - 1 - boxes[:, 0]
                boxes[:, 2] = xmax
                boxes[:, 0] = xmin
            if keypoints.shape[0] != 0:
                keypoints[:, :, 0] = width - 1.0 - keypoints[:, :, 0]

            target["boxes"] = boxes
            target["keypoints"] = keypoints
            return {'image': cv2.flip(img, 1),'target': target}
        return {'image': img,'target': target}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            height, width, _ = img.shape
            boxes = target["boxes"]
            keypoints = target["keypoints"]
            if boxes.shape[0] != 0:
                ymin = height -1 - boxes[:, 3]
                ymax = height -1 - boxes[:, 1]
                boxes[:, 3] = ymax
                boxes[:, 1] = ymin

            if keypoints.shape[0] != 0:
                keypoints[:, :, 1] = height - 1.0 - keypoints[:, :, 1]

            target["boxes"] = boxes
            target["keypoints"] = keypoints
            return {'image': cv2.flip(img, 0), 'target': target}
        return {'image': img, 'target': target}


class RescaleRelative(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        img, target = sample['image'], sample['target']


        return {'image': img, 'target': target}



class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, sample):
        if isinstance(sample, list):
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
            return sample
        else:
            img, target = sample['image'], sample['target']
            anno = target["annotations"]
            anno = [obj for obj in anno if obj["category_id"] in self.categories]
            if not self.remap:
                target["annotations"] = anno
                return {'image': img, 'target': target}
            anno = copy.deepcopy(anno)
            for obj in anno:
                obj["category_id"] = self.categories.index(obj["category_id"])
            target["annotations"] = anno
            return {'image': img, 'target': target}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        # mask = mask.any(dim=2)
        mask = np.any(mask, 2)
        masks.append(mask)
    if masks:
        # masks = torch.stack(masks, dim=0)
        masks = np.stack(masks, 0)
    else:
        # masks = torch.zeros((0, height, width), dtype=torch.uint8)
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, use_mask=False, use_keypoints=False):
        self.use_mask = use_mask
        self.use_keypoints = use_keypoints

    def __call__(self, sample):
        if isinstance(sample, list):
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
            return sample
        else:
            img, target = sample['image'], sample['target']
            h, w, _ = img.shape

            image_id = target["image_id"]
            anno = target["annotations"]

            anno = [obj for obj in anno if obj['iscrowd'] == 0]

            boxes = [obj["bbox"] for obj in anno]
            # guard against no boxes via resizing
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]  # xywh ==> xyxy
            boxes = clip_boxes_to_image(boxes, [h, w])

            labels = [obj["category_id"] for obj in anno]
            labels = torch.tensor(labels, dtype=torch.int64)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]

            target = {}
            target["image_id"] = torch.tensor([image_id])
            target["height"] = torch.tensor(h)
            target["width"] = torch.tensor(w)
            target["boxes"] = boxes
            target["labels"] = labels
            if self.use_mask:
                segmentations = [obj["segmentation"] for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                target["masks"] = masks[keep]
            if self.use_keypoints:
                keypoints = None
                if anno and "keypoints" in anno[0]:
                    keypoints = [obj["keypoints"] for obj in anno]
                    keypoints = np.array(keypoints, dtype=np.float32).reshape(-1, 51)
                    # keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                    num_keypoints = keypoints.shape[0]
                    if num_keypoints:
                        keypoints = keypoints.reshape(num_keypoints, -1, 3)
                if keypoints is not None:
                    keypoints = keypoints[keep]
                    target["keypoints"] = keypoints

            # for conversion to coco api
            # area = torch.tensor([obj["area"] for obj in anno])
            # iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

            # target["area"] = area
            # target["iscrowd"] = iscrowd
            return {'image': img, 'target': target}
