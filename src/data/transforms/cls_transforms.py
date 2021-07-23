# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/17 8:42
# @Author : liumin
# @File : cls_transforms.py

import random
from numbers import Number
import cv2
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
import numpy as np


__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Resize', 'RandomScale', 'RandomCrop', 'CenterCrop',
        'RandomRotate', 'RandomTranslation',
        'ColorJitter', 'RandomGaussianBlur',
        'Normalize', 'DeNormalize', 'ToTensor',
        'RGB2BGR', 'BGR2RGB']


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