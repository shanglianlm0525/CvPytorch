# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/18 15:35
# @Author : liumin
# @File : keypoint_transforms.py
import copy
import math
import random
import warnings

import cv2
import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from pycocotools import mask as coco_mask

__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Normalize', 'ToTensor',
        'RandomHorizontalFlip', 'RandomVerticalFlip',
        'Resize', 'RandomResizedCrop',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']


def remove_small_boxes(boxes, min_size):
    """
       Remove boxes which contains at least one side smaller than min_size.

       Args:
           boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
               with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
           min_size (float): minimum size

       Returns:
           Tensor[K]: indices of the boxes that have both sides
           larger than min_size
       """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = np.where(keep)[0]
    return keep

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

        if target.__contains__("boxes"):
            target["boxes"] = torch.from_numpy(target["boxes"])
        if target.__contains__("masks"):
            mask = target["masks"]
            mask = mask.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            mask = np.ascontiguousarray(mask)
            target["masks"] = torch.from_numpy(mask)
        # if target.__contains__("keypoints"):
        #     target["keypoints"] = torch.from_numpy(target["keypoints"])
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

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                if boxes.shape[0] != 0:
                    xmin = width - 1 - boxes[:, 2]
                    xmax = width - 1 - boxes[:, 0]
                    boxes[:, 2] = xmax
                    boxes[:, 0] = xmin
                target["boxes"] = boxes

            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 0] = width - 1.0 - keypoints[:, :, 0]
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

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                if boxes.shape[0] != 0:
                    ymin = height - 1 - boxes[:, 3]
                    ymax = height - 1 - boxes[:, 1]
                    boxes[:, 3] = ymax
                    boxes[:, 1] = ymin
                target["boxes"] = boxes

            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 1] = height - 1.0 - keypoints[:, :, 1]
                target["keypoints"] = keypoints

            return {'image': cv2.flip(img, 0), 'target': target}
        return {'image': img, 'target': target}


class Resize(object):
    """Resize the input img to the given size."""

    def __init__(self, size, keep_ratio=True, scaleup=True, fill=[128, 128, 128]):
        self.size = size if isinstance(size, list) else [size, size]
        self.keep_ratio = keep_ratio
        self.scaleup = scaleup # only valid when the keep_ratio is True
        self.fill = fill

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        height, width, _ = img.shape

        if self.keep_ratio:
            scale = min(self.size[0] / height, self.size[1] / width)
            if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
                scale = min(scale, 1.0)
            oh, ow = int(round(height * scale)), int(round(width * scale))
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides

            if (height != oh) or (width != ow):
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                boxes[:, 1::2] = boxes[:, 1::2] * scale + top
                boxes[:, 0::2] = boxes[:, 0::2] * scale + left
                target["boxes"] = boxes

            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 0] = keypoints[:, :, 0] * scale + top
                    keypoints[:, :, 1] = keypoints[:, :, 1] * scale + left
                target["keypoints"] = keypoints

            target["pads"] = torch.tensor([top, left], dtype=torch.float)
            target["scales"] = torch.tensor([scale, scale], dtype=torch.float)
            return {'image': img, 'target': target}
        else:
            scale_h, scale_w = self.size[0] / height, self.size[1] / width
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                target["boxes"] = boxes

            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 0] *= scale_w
                    keypoints[:, :, 1] *= scale_h
                target["keypoints"] = keypoints

            target["pads"] = torch.tensor([0, 0], dtype=torch.float)
            target["scales"] = torch.tensor([scale_h, scale_w], dtype=torch.float)
            return {'image': img, 'target': target}


class RandomResizedCrop1(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), keep_ratio=True, fill=[128, 128, 128], min_size = 3):
        super().__init__()
        self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.keep_ratio = keep_ratio
        self.fill = fill
        self.min_size = min_size

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width, _ = img.shape
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        labels = target["labels"]
        keypoints = target["keypoints"]
        height, width, _ = img.shape
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if self.keep_ratio:
            # crop (top, left, height, width)
            img = img[i:(i + h), j:(j + w), :]
            boxes -= np.array([j, i, j, i])
            boxes = clip_boxes_to_image(boxes, [h, w])
            if keypoints.shape[0] != 0:
                keypoints[:, :, 0] -= j
                keypoints[:, :, 1] -= i

            # resize
            scale = min(self.size[0] / h, self.size[1] / w)
            ow = int(w * scale)
            oh = int(h * scale)
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            # pad,  left, top, right and bottom
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides
            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border
            boxes[:, 1::2] = boxes[:, 1::2] * scale + top
            boxes[:, 0::2] = boxes[:, 0::2] * scale + left
            if keypoints.shape[0] != 0:
                keypoints[:, :, 0] = keypoints[:, :, 0] * scale + top
                keypoints[:, :, 1] = keypoints[:, :, 1] * scale + left

            keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
            if keep.shape[0] < 1:
                img, target = sample['image'], sample['target']
                boxes = target["boxes"]
                scale = min(self.size[0] / h, self.size[1] / w)
                ow = int(w * scale)
                oh = int(h * scale)
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
                padh /= 2
                padw /= 2  # divide padding into 2 sides
                top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
                left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)
                boxes[:, 1::2] = boxes[:, 1::2] * scale + top
                boxes[:, 0::2] = boxes[:, 0::2] * scale + left
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 0] = keypoints[:, :, 0] * scale + top
                    keypoints[:, :, 1] = keypoints[:, :, 1] * scale + left

                target["boxes"] = boxes
                target["keypoints"] = keypoints
                return {'image': img, 'target': target}

            target["labels"] = labels[keep]
            target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()
            target["keypoints"] = keypoints
            return {'image': img, 'target': target}
        else:
            # crop
            img = img[i:(i + h), j:(j + w), :]
            boxes -= [j, i, j, i]
            boxes = clip_boxes_to_image(boxes, [h, w])
            if keypoints.shape[0] != 0:
                keypoints[:, :, 0] -= j
                keypoints[:, :, 1] -= i

            # resize
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            scale_h, scale_w = self.size[0] / h, self.size[1] / w
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            if keypoints.shape[0] != 0:
                keypoints[:, :, 0] *= scale_w
                keypoints[:, :, 1] *= scale_h

            keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
            if keep.shape[0] < 1:
                img, target = sample['image'], sample['target']
                boxes = target["boxes"]
                img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
                scale_h, scale_w = self.size[0] / h, self.size[1] / w
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h
                if keypoints.shape[0] != 0:
                    keypoints[:, :, 0] *= scale_w
                    keypoints[:, :, 1] *= scale_h
                target["boxes"] = boxes
                target["keypoints"] = keypoints
                return {'image': img, 'target': target}

            target["labels"] = labels[keep]
            target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()
            target["keypoints"] = keypoints
            return {'image': img, 'target': target}


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), keep_ratio=True, fill=[128, 128, 128], min_size = 3):
        super().__init__()
        self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.keep_ratio = keep_ratio
        self.fill = fill
        self.min_size = min_size

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        height, width, _ = img.shape
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w


    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img, target = sample['image'], sample['target']

        height, width, _ = img.shape
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if self.keep_ratio:
            # crop (top, left, height, width)
            img = img[i:(i + h), j:(j + w), :]
            # resize
            scale = min(self.size[0] / h, self.size[1] / w)
            ow = int(w * scale)
            oh = int(h * scale)
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            # pad,  left, top, right and bottom
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides
            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                labels = target["labels"]
                # crop
                boxes -= np.array([j, i, j, i])
                boxes = clip_boxes_to_image(boxes, [h, w])
                # resize
                boxes[:, 1::2] = boxes[:, 1::2] * scale + top
                boxes[:, 0::2] = boxes[:, 0::2] * scale + left

                keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
                if keep.shape[0] < 1:
                    img, target = sample['image'], sample['target']
                    boxes = target["boxes"]
                    scale = min(self.size[0] / h, self.size[1] / w)
                    ow = int(w * scale)
                    oh = int(h * scale)
                    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
                    padh /= 2
                    padw /= 2  # divide padding into 2 sides
                    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
                    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
                    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)

                    boxes[:, 1::2] = boxes[:, 1::2] * scale + top
                    boxes[:, 0::2] = boxes[:, 0::2] * scale + left

                    target["boxes"] = boxes
                    return {'image': img, 'target': target}
                target["labels"] = labels[keep]
                target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()


            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]

                if keypoints.shape[0] != 0:
                    # crop
                    keypoints[:, :, 0] -= j
                    keypoints[:, :, 1] -= i
                    keypoints[np.logical_or(keypoints[:, :, 0] < 0, keypoints[:, :, 1] < 0), 2] = 0 # set to invidi
                    # resize
                    keypoints[:, :, 0] = keypoints[:, :, 0] * scale + top
                    keypoints[:, :, 1] = keypoints[:, :, 1] * scale + left

                if np.sum(np.sum(keypoints[:,:,2])) < 1:
                    img, target = sample['image'], sample['target']
                    scale = min(self.size[0] / h, self.size[1] / w)
                    ow = int(w * scale)
                    oh = int(h * scale)
                    img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
                    padh /= 2
                    padw /= 2  # divide padding into 2 sides
                    top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
                    left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
                    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)

                    if keypoints.shape[0] != 0:
                        keypoints[:, :, 0] = keypoints[:, :, 0] * scale + top
                        keypoints[:, :, 1] = keypoints[:, :, 1] * scale + left
                    target["keypoints"] = keypoints
                    return {'image': img, 'target': target}
                target["keypoints"] = keypoints

            return {'image': img, 'target': target}
        else:
            # crop
            img = img[i:(i + h), j:(j + w), :]

            # resize
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            scale_h, scale_w = self.size[0] / h, self.size[1] / w

            if target.__contains__("boxes"):
                boxes = target["boxes"]
                labels = target["labels"]
                # crop
                boxes -= [j, i, j, i]
                boxes = clip_boxes_to_image(boxes, [h, w])
                # resize
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h

                keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
                if keep.shape[0] < 1:
                    img, target = sample['image'], sample['target']
                    boxes = target["boxes"]
                    img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
                    scale_h, scale_w = self.size[0] / h, self.size[1] / w
                    boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                    boxes[:, 1::2] = boxes[:, 1::2] * scale_h
                    target["boxes"] = boxes
                    return {'image': img, 'target': target}

                target["labels"] = labels[keep]
                target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()

            if target.__contains__("keypoints"):
                keypoints = target["keypoints"]

                if keypoints.shape[0] != 0:
                    # crop
                    keypoints[:, :, 0] -= j
                    keypoints[:, :, 1] -= i
                    keypoints[np.logical_or(keypoints[:, :, 0] < 0, keypoints[:, :, 1] < 0), 2] = 0 # set to invidi
                    # resize
                    keypoints[:, :, 0] *= scale_w
                    keypoints[:, :, 1] *= scale_h

                if np.sum(np.sum(keypoints[:,:,2])) < 1:
                    img, target = sample['image'], sample['target']
                    img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
                    scale_h, scale_w = self.size[0] / h, self.size[1] / w
                    if keypoints.shape[0] != 0:
                        keypoints[:, :, 0] *= scale_w
                        keypoints[:, :, 1] *= scale_h
                    target["keypoints"] = keypoints
                    return {'image': img, 'target': target}
                target["keypoints"] = keypoints

            return {'image': img, 'target': target}


class CropWithFactor(object):
    def __init__(self, size=None, factor=32, is_ceil=True):
        self.size = size
        self.factor = factor
        self.is_ceil = is_ceil

    def __call__(self, sample):
        img, target = sample['image'], sample['target']

        h, w, _ = img.shape
        max_size, min_size = (h, w) if h > w else (w, h)
        img_scale = float(self.size) / min_size
        img = cv2.resize(img, None, fx=img_scale, fy=img_scale)

        h, w, c = img.shape
        new_h = self._factor_closest(h)
        new_w = self._factor_closest(w)
        img_croped = np.zeros([new_h, new_w, c], dtype=img.dtype)
        img_croped[0:h, 0:w, :] = img

        if target.__contains__("keypoints"):
            keypoints = target["keypoints"]
            if keypoints.shape[0] != 0:
                keypoints[:, :, :2] *= img_scale
            target["keypoints"] = keypoints

        return {'image': img_croped, 'target': target}

    def _factor_closest(self, num):
        return int(np.ceil(float(num) / self.factor) if self.is_ceil else np.floor(float(num) / self.factor)) * self.factor


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
    def __init__(self, use_box=False, use_mask=False, use_keypoints=False):
        self.use_box = use_box
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

            target = {}
            target["image_id"] = torch.tensor([image_id])
            target["height"] = torch.tensor(h)
            target["width"] = torch.tensor(w)

            keep = np.ones((len(anno)), np.bool)
            if self.use_box:
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
