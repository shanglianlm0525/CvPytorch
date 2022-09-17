# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/17 8:42
# @Author : liumin
# @File : cls_transforms.py

import warnings
import math
import random
from numbers import Number
import cv2
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numpy as np
from collections.abc import Sequence


__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomResizedCrop',
        'Resize', 'RandomScale', 'RandomCrop', 'CenterCrop',
        'RandomRotate', 'RandomTranslation',
        'ColorJitter', 'RandomGaussianBlur',
        'Normalize', 'DeNormalize', 'ToTensor',
        'RGB2BGR', 'BGR2RGB']


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
    boxes[..., 0::2] = boxes[..., 0::2].clip(min=0, max=width-1)
    boxes[..., 1::2] = boxes[..., 1::2].clip(min=0, max=height-1)
    return boxes


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


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


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img, target = sample['image'], sample['target']
        img = F.to_tensor(img.astype(np.uint8))
        target = torch.tensor(target)
        return {'image': img, 'target': target}


class Resize(object):
    def __init__(self, size): # size: (h, w)
        self.size = (size, size) if isinstance(size, int) else (size)

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = cv2.resize(img, tuple(self.size[::-1]), interpolation=cv2.INTER_LINEAR)
        return {'image': img,'target': target}


class RandomResizedCrop(object):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), keep_ratio=True, fill=[0, 0, 0], min_size = 3):
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
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

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
        # height, width, _ = img.shape
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # crop (top, left, height, width)
        img = img[i:(i + h), j:(j + w), :]
        if self.keep_ratio:
            # resize
            scale = min(self.size[0] / h, self.size[1] / w)
            oh, ow = int(round(h * scale)), int(round(w * scale))
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides

            if (h != oh) or (w != ow):
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border
            return {'image': img, 'target': target }
        else:
            # resize
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            return {'image': img, 'target': target }


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.flip(img, 1)
        return {'image': img,'target': target}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.flip(img, 0)
        return {'image': img,'target': target}


class RandomRotate(object):
    def __init__(self, p=0.5, degree=(0,0)):
        self.p = p
        self.degree = (-1*degree, degree) if isinstance(degree, int) else degree

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            angle = self.degree[0] + (self.degree[1] - self.degree[0]) * random.random()
            h, w, _ = img.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)
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


class RandomTranslation(object):
    def __init__(self, p=0.5, pixel=(2,2)): # (h, w)
        self.p = p
        if isinstance(pixel, Number):
            self.transX = random.randint(pixel, pixel)
            self.transY = random.randint(pixel, pixel)
        else:
            self.transX = random.randint(pixel[1], pixel[1])
            self.transY = random.randint(pixel[0], pixel[0])

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        # Random translation 0-2 pixels (fill rest with padding
        if random.random() < self.p:
            img = cv2.copyMakeBorder(img, self.transX, 0, self.transY, 0, cv2.BORDER_CONSTANT, (0, 0, 0)) # top, bottom, left, right
            h, w, _ = img.shape
            img = img[:h - self.transY, :w - self.transX]
        return {'image': img,'target': target}


class RandomCrop(object):
    """Crops the given ndarray image (H*W*C or H*W).
        Args:
            size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is made.
        """
    def __init__(self, size):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample): # [h, w]
        img, target = sample['image'], sample['target']
        h, w,_ = img.shape
        pad_h = max(self.size[0] - h, 0)
        pad_w = max(self.size[1] - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                       cv2.BORDER_CONSTANT, value=0)
        h, w, _ = img.shape
        h_off = random.randint(0, h - self.size[0])
        w_off = random.randint(0, w - self.size[1])
        img = img[h_off:h_off + self.size[0], w_off:w_off + self.size[1]]
        return {'image': img,'target': target}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w,_ = img.shape
        pad_h = max(self.size[0] - h, 0)
        pad_w = max(self.size[1] - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            img = cv2.copyMakeBorder(img, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                                     cv2.BORDER_CONSTANT, value=0)
        h, w,_ = img.shape
        h_off = int((h - self.size[0]) / 2)
        w_off = int((w - self.size[1]) / 2)
        img = img[h_off:h_off + self.size[0], w_off:w_off + self.size[1]]
        return {'image': img, 'target': target}


class ColorJitter(object):
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = Image.fromarray(img)
            img = T.ColorJitter(self.brightness, self.contrast, self.saturation,self.hue)(img)
            img = np.asarray(img, dtype=np.int32)
        return {'image': img, 'target': target}


class RandomScale(object):
    def __init__(self, size): # (short edge)
        self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        short_size = random.randint(int(self.size * 0.5), int(self.size * 2.0))
        h, w, _ = img.shape
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
            return {'image': cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR), 'target': target}
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
            return {'image': cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR), 'target': target}


class Scale(object):
    def __init__(self, size): # (h, w)
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': img,'target': target}
        if w > h:
            ow = self.size[1]
            oh = int(self.size[0] * h / w)
            return {'image': cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR), 'target': target}
        else:
            oh = self.size[0]
            ow = int(self.size[1] * w / h)
            return {'image': cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR), 'target': target}


class FixScaleCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape
        short_size = min(self.size)
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)

        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        # center crop
        h, w, _ = img.shape
        h_off = int((h - self.size[0]) / 2)
        w_off = int((w - self.size[1]) / 2)
        img = img[h_off:h_off + self.size[0], w_off:w_off + self.size[1]]
        return {'image': img,'target': target}


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