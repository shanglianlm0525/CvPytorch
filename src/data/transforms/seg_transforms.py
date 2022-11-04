# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/18 18:34
# @Author : liumin
# @File : seg_transforms.py

import collections
import copy
import math
import numbers
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from pycocotools import mask as coco_mask
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from PIL import Image, ImageOps, ImageEnhance

__all__ = ['Compose', 'ToTensor', 'Normalize',
           'RandomHorizontalFlip', 'RandomVerticalFlip', 'RandomDiagonalFlip',
           'Resize', 'RandomScaleResize', 'RandomScaleCrop',
           'RandomCrop', 'CenterCrop', 'Pad', 'RandomRotate',
           'ColorJitter', 'PhotoMetricDistortion','GaussianBlur', 'MedianBlur',  'RandomGrayscale',
           'RandAugment', 'Lambda', 'Encrypt',
           'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']


def pad_img_and_target(img, target, size, fill, ignore_label, borderType=cv2.BORDER_CONSTANT):
    h, w = img.shape[:2]
    if h < size[0] or w < size[1]:
        # padding
        padh, padw = max(size[0] - h, 0), max(size[1] - w, 0)  # wh padding
        padh /= 2
        padw /= 2  # divide padding into 2 sides
        top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
        left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, value=fill)  # add border
        target = cv2.copyMakeBorder(target, top, bottom, left, right, borderType, value=ignore_label)
    return img, target


def resize_img_and_target(img, target, size, keep_ratio=True,
                          interpolation_img=cv2.INTER_LINEAR, interpolation_target=cv2.INTER_NEAREST):
    h, w = img.shape[:2]
    if keep_ratio:
        # random scale (short edge)
        # resize so that the max edge no longer than max(h, w), short edge no longer than min(h,w)
        # without changing the aspect ratio
        ratio = min(size[0] / h, size[1] / w)
        oh, ow = int(round(h * ratio)), int(round(w * ratio))

        if (h != oh) or (w != ow):
            img = cv2.resize(img, (ow, oh), interpolation=interpolation_img)
            target = cv2.resize(target, (ow, oh), interpolation=interpolation_target)
    else:
        img = cv2.resize(img, size[::-1], interpolation=interpolation_img)
        target = cv2.resize(target, size[::-1], interpolation=interpolation_target)

    return img, target


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, normalize=True, to_rgb=True, target_type='uint8'):
        self.normalize = normalize
        self.to_rgb = to_rgb
        self.target_type = target_type

    def __call__(self, sample):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to tensor.
            target (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        img, target = sample['image'], sample['target']
        if self.to_rgb:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # inplace
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        else:
            img = img.transpose((2, 0, 1))  # HWC to CHW
        img = np.ascontiguousarray(img)

        '''
        if len(target.shape) < 3:
            target = target[None, ...]
        '''

        if self.normalize:
            return {'image': torch.from_numpy(img.astype(np.float32)).div_(255.0),
                    'target': torch.from_numpy(target.astype(self.target_type))}
        else:
            return {'image': torch.from_numpy(img.astype(np.float32)),
                    'target': torch.from_numpy(target.astype(self.target_type))}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        img, target = sample['image'], sample['target']
        return {'image': TF.normalize(img, self.mean, self.std), 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': np.flip(img, 1), 'target': np.flip(target, 1)}
        return {'image': img, 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': np.flip(img, 0), 'target': np.flip(target, 0)}
        return {'image': img, 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomDiagonalFlip(object):
    """Diagonal flip the given Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': np.flip(img, (0, 1)), 'target': np.flip(target, (0, 1))}
        return {'image': img, 'target': target}
    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomScaleCrop(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size=None, scale=None, keep_ratio=True, pad_if_needed=False, fill=0, ignore_label=255):
        assert any([size, scale]) , 'size and scale are not all None'
        self.size = size
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label

    def get_crop_bbox(self, img, crop_size):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape

        # random_scale
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
            new_size = int(round(h * scale)), int(round(w * scale))
        else:
            new_size = self.size

        img, target = resize_img_and_target(img, target, new_size, self.keep_ratio)

        # crop
        crop_bbox = self.get_crop_bbox(img, self.size)
        # crop the image
        img = self.crop(img, crop_bbox)
        target = self.crop(target, crop_bbox)

        if self.pad_if_needed:
            img, target = pad_img_and_target(img, target, self.size, self.fill, self.ignore_label)

        return {'image': img, 'target': target}


class RandomScaleResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """
    def __init__(self, size=None, scale=None, keep_ratio=True, pad_if_needed=False, fill=0, ignore_label=255):
        assert any([size, scale]) , 'size and scale are not all None'
        self.size = size
        self.scale = scale
        self.keep_ratio = keep_ratio
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape

        # random_scale
        if self.scale is not None:
            scale = random.uniform(self.scale[0], self.scale[1])
            if self.size is None:
                new_size = int(round(h * scale)), int(round(w * scale))
            else:
                new_size = int(round(self.size[0] * scale)), int(round(self.size[1] * scale))
        else:
            new_size = self.size

        img, target = resize_img_and_target(img, target, new_size, self.keep_ratio)

        if self.keep_ratio and self.pad_if_needed:
            img, target = pad_img_and_target(img, target, self.size, self.fill, self.ignore_label)

        return {'image': img, 'target': target}


class Resize(object):
    """Resize the input img to the given size."""

    def __init__(self, size=None, keep_ratio=True, pad_if_needed=False, fill=0, ignore_label=255):
        assert size is not None, 'size is not None'
        self.size = size
        self.keep_ratio = keep_ratio
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape

        img, target = resize_img_and_target(img, target, self.size, self.keep_ratio)

        if self.keep_ratio and self.pad_if_needed:
            img, target = pad_img_and_target(img, target, self.size, self.fill, self.ignore_label)

        return {'image': img, 'target': target}


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, pad_if_needed=False, fill=0, ignore_label=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label

    def get_crop_bbox(self, img, crop_size):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, sample):
        img, target = sample['image'], sample['target']

        crop_bbox = self.get_crop_bbox(img, self.size)

        # crop the image
        img = self.crop(img, crop_bbox)
        target = self.crop(target, crop_bbox)

        if self.pad_if_needed:
            img, target = pad_img_and_target(img, target, self.size, self.fill, self.ignore_label)

        return { 'image': img, 'target': target }

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size, pad_if_needed=False, fill=0, ignore_label=255):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label

    def get_crop_bbox(self, img, crop_size):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)

        crop_y1, crop_x1 = margin_h // 2, margin_w // 2
        crop_y2, crop_x2 = crop_y1 + crop_size[0], crop_x1 + crop_size[1]
        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img, target = sample['image'], sample['target']

        crop_bbox = self.get_crop_bbox(img, self.size)

        # crop the image
        img = self.crop(img, crop_bbox)
        target = self.crop(target, crop_bbox)

        if self.pad_if_needed:
            img, target = pad_img_and_target(img, target, self.size, self.fill, self.ignore_label)

        return {'image': img, 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Pad(object):
    def __init__(self, size=None, diviser=None, fill=0, ignore_label=255):
        self.size = size
        self.diviser = diviser
        self.fill = fill
        self.ignore_label = ignore_label
        # only one of size and size_divisor should be valid
        assert size is not None or diviser is not None
        assert size is None or diviser is None

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w = img.shape[:2]

        if self.size is not None:
            padh, padw = max(self.size[0] - h, 0), max(self.size[1] - w, 0)  # wh padding
        elif self.diviser is not None:
            padh = (h // self.diviser + 1) * self.diviser - h if h % self.diviser != 0 else 0
            padw = (w // self.diviser + 1) * self.diviser - w if w % self.diviser != 0 else 0
        else:
            raise NotImplementedError

        padh /= 2
        padw /= 2  # divide padding into 2 sides
        top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
        left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border
        target = cv2.copyMakeBorder(target, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.ignore_label)

        return {'image': img, 'target': target}


class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(0, 1):
            return self.convert(img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(0, 1):
            return self.convert(img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))

            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(0, 1):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, sample):
        """Call function to perform photometric distortion on images."""
        img, target = sample['image'], sample['target']

        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        return {'image': img, 'target': target}

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            image = Image.fromarray(img.astype(np.uint8))
            image = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(image)
            return {'image': np.asarray(image), 'target': target}
        return {'image': img, 'target': target}

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return {'image': img, 'target': target}


class GaussianBlur(object):
    def __init__(self, p=0.01, ksize=[3, 5, 7]):
        self.p = p
        self.ksize = ksize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            ksize = int(random.choice(self.ksize))
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        return {'image': img, 'target': target}


class MedianBlur(object):
    def __init__(self, p=0.01, ksize=[3, 5, 7]):
        self.p = p
        self.ksize = ksize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.medianBlur(img, int(random.choice(self.ksize)))
        return {'image': img, 'target': target}


class RandomRotate(torch.nn.Module):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, p, degree, center=None, auto_bound=False, fill=0, ignore_label=255):
        super().__init__()
        self.prpob = p
        assert p >= 0 and p <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        if center is not None and auto_bound:
            raise ValueError('`auto_bound` conflicts with `center`')
        self.fill = fill
        self.ignore_label = ignore_label
        self.center = center
        self.auto_bound = auto_bound

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        img, target = sample['image'], sample['target']

        if random.random() < self.p:
            angle = random.uniform(self.degree[0], self.degree[1])
            h, w = img.shape[:2]
            if self.center is None:
                self.center = ((w - 1) * 0.5, (h - 1) * 0.5)

            matrix = cv2.getRotationMatrix2D(self.center, -angle, 1.0)
            if self.auto_bound:
                cos = np.abs(matrix[0, 0])
                sin = np.abs(matrix[0, 1])
                new_w = h * sin + w * cos
                new_h = h * cos + w * sin
                matrix[0, 2] += (new_w - w) * 0.5
                matrix[1, 2] += (new_h - h) * 0.5
                w = int(np.round(new_w))
                h = int(np.round(new_h))
            img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderValue=self.fill)
            target = cv2.warpAffine(target, matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=self.ignore_label)

        return {'image': img, 'target': target}


# Minimum value for posterize (0 in EfficientNet implementation)
POSTERIZE_MIN = 1

# Parameters for affine warping and rotation
WARP_PARAMS = {"fillcolor": (128, 128, 128), "resample": Image.BILINEAR}


def affine_warp(im, data):
    """Applies affine transform to image."""
    return im.transform(im.size, Image.AFFINE, data, **WARP_PARAMS)


OP_FUNCTIONS = {
    # Each op takes an image x and a level v and returns an augmented image.
    "auto_contrast": lambda x, _: ImageOps.autocontrast(x),
    "equalize": lambda x, _: ImageOps.equalize(x),
    "invert": lambda x, _: ImageOps.invert(x),
    "rotate": lambda x, v: x.rotate(v, **WARP_PARAMS),
    "posterize": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, int(v))),
    "posterize_inc": lambda x, v: ImageOps.posterize(x, max(POSTERIZE_MIN, 4 - int(v))),
    "solarize": lambda x, v: x.point(lambda i: i if i < int(v) else 255 - i),
    "solarize_inc": lambda x, v: x.point(lambda i: i if i < 256 - v else 255 - i),
    "solarize_add": lambda x, v: x.point(lambda i: min(255, v + i) if i < 128 else i),
    "color": lambda x, v: ImageEnhance.Color(x).enhance(v),
    "contrast": lambda x, v: ImageEnhance.Contrast(x).enhance(v),
    "brightness": lambda x, v: ImageEnhance.Brightness(x).enhance(v),
    "sharpness": lambda x, v: ImageEnhance.Sharpness(x).enhance(v),
    "color_inc": lambda x, v: ImageEnhance.Color(x).enhance(1 + v),
    "contrast_inc": lambda x, v: ImageEnhance.Contrast(x).enhance(1 + v),
    "brightness_inc": lambda x, v: ImageEnhance.Brightness(x).enhance(1 + v),
    "sharpness_inc": lambda x, v: ImageEnhance.Sharpness(x).enhance(1 + v),
    "shear_x": lambda x, v: affine_warp(x, (1, v, 0, 0, 1, 0)),
    "shear_y": lambda x, v: affine_warp(x, (1, 0, 0, v, 1, 0)),
    "trans_x": lambda x, v: affine_warp(x, (1, 0, v * x.size[0], 0, 1, 0)),
    "trans_y": lambda x, v: affine_warp(x, (1, 0, 0, 0, 1, v * x.size[1])),
}

affine_ops = [
    "rotate", "shear_x", "shear_y", "trans_x", "trans_y"
]

OP_RANGES = {
    # Ranges for each op in the form of a (min, max, negate).
    "auto_contrast": (0, 1, False),
    "equalize": (0, 1, False),
    "invert": (0, 1, False),
    "rotate": (0.0, 30.0, True),
    "posterize": (0, 4, False),
    "posterize_inc": (0, 4, False),
    "solarize": (0, 256, False),
    "solarize_inc": (0, 256, False),
    "solarize_add": (0, 110, False),
    "color": (0.1, 1.9, False),
    "contrast": (0.1, 1.9, False),
    "brightness": (0.1, 1.9, False),
    "sharpness": (0.1, 1.9, False),
    "color_inc": (0, 0.9, True),
    "contrast_inc": (0, 0.9, True),
    "brightness_inc": (0, 0.9, True),
    "sharpness_inc": (0, 0.9, True),
    "shear_x": (0.0, 0.3, True),
    "shear_y": (0.0, 0.3, True),
    "trans_x": (0.0, 0.45, True),
    "trans_y": (0.0, 0.45, True),
}

RANDAUG_OPS = [
    # RandAugment list of operations using "increasing" transforms.
    "auto_contrast",
    "equalize",
    # "invert",
    "rotate",
    "posterize_inc",
    "solarize_inc",
    "solarize_add",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
    "shear_x",
    "shear_y",
    "trans_x",
    "trans_y",
]

RANDAUG_OPS_REDUCED = [
    "auto_contrast",
    "equalize",
    "rotate",
    "color_inc",
    "contrast_inc",
    "brightness_inc",
    "sharpness_inc",
]


class RandAugment(object):
    """
        RandAugment: Practical automated data augmentation with a reduced search space
        https://arxiv.org/pdf/1909.13719.pdf
    """

    def __init__(self, p, n_ops, magnitude, ops="reduced", fill=(0, 0, 0), ignore_value=255):
        super(RandAugment, self).__init__()
        assert 0 <= magnitude <= 1
        self.p = p
        self.n_ops = n_ops
        self.magnitude = magnitude
        self.fill = fill
        self.ignore_value = ignore_value
        ops = ops if ops else RANDAUG_OPS
        if ops == "full":
            ops = RANDAUG_OPS
        elif ops == "reduced":
            ops = RANDAUG_OPS_REDUCED
        else:
            raise NotImplementedError()
        self.ops = ops

    def __call__(self, sample):
        img, target = sample['image'], sample['target']

        for op in random.sample(self.ops, int(self.n_ops)):
            if self.p < 1 and random.random() > self.p:
                continue
            # trans numpy to Image
            img = Image.fromarray(img.astype(np.uint8))
            target = Image.fromarray(target.astype(np.uint8))

            min_v, max_v, negate = OP_RANGES[op]
            v = self.magnitude * (max_v - min_v) + min_v
            v = -v if negate and random.random() > 0.5 else v
            WARP_PARAMS["fillcolor"] = self.fill
            WARP_PARAMS["resample"] = Image.BILINEAR
            img = OP_FUNCTIONS[op](img, v)
            if op in affine_ops:
                WARP_PARAMS["fillcolor"] = self.ignore_value
                WARP_PARAMS["resample"] = Image.NEAREST
                target = OP_FUNCTIONS[op](target, v)

            # trans Image to numpy
            img = np.asarray(img)
            target = np.asarray(target)

        return {'image': img, 'target': target}


class Encrypt(object):
    def __init__(self, scale_factor):  # size: (h, w)
        self.scale_factor = (scale_factor, scale_factor) if isinstance(scale_factor, int) else scale_factor

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape
        target = target.resize((w // self.scale_factor[1], h // self.scale_factor[0]))
        return {'image': img, 'target': target}


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

####------------------- COCO Dataset ---------------#####

class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
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
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape
        anno = target["annotations"]
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=torch.uint8)
        return {'image': img, 'target': target}

####------------------- COCO Dataset ---------------#####

