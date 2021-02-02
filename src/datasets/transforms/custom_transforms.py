# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/9/25 14:30
# @Author : liumin
# @File : custom_transforms.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision import transforms as T
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from numbers import Number

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        assert img.size == target.size
        for t in self.transforms:
            sample = t(sample)
        return sample


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.),inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        img,target = sample['image'], sample['target']
        img = TF.normalize(img,self.mean,self.std,self.inplace)
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

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        target = np.array(target).astype(np.float32)
        if len(target.shape) == 2:
            target = np.expand_dims(target, axis=0)

        img = torch.from_numpy(img).float() / 255.0
        target = torch.from_numpy(target).float()
        return {'image': img, 'target': target}


class Encrypt(object):
    def __init__(self, down_size): # size: (w, h)
        self.down_size = (down_size, down_size) if isinstance(down_size, int) else down_size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        target = target.resize((int(img.size(0) / self.down_size[0]), (int(img.size(1) / self.down_size[1]))), Image.NEAREST)
        return {'image': img,'target': target}


class Resize(object):
    def __init__(self, size): # size: (w, h)
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        assert img.size == target.size
        img = img.resize(self.size, Image.BILINEAR)
        target = target.resize(self.size, Image.NEAREST)
        return {'image': img,'target': target}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)
        return {'image': img,'target': target}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            target = target.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': img,'target': target}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        target = target.rotate(rotate_degree, Image.NEAREST)
        return {'image': img,'target': target}


class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        return {'image': img,'target': target}


class RandomTranslation(object):
    def __init__(self, pixel, padding=255):
        if isinstance(pixel, Number):
            self.transX = random.randint(-pixel, pixel)
            self.transY = random.randint(-pixel, pixel)
        else:
            self.transX = random.randint(-pixel[0], pixel[0])
            self.transY = random.randint(-pixel[1], pixel[1])
        self.padding = padding

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        # Random translation 0-2 pixels (fill rest with padding
        img = ImageOps.expand(img, border=(self.transX, self.transY, 0, 0), fill=0)
        target = ImageOps.expand(target, border=(self.transX, self.transY, 0, 0), fill=self.padding)  # pad label filling with 255
        img = img.crop((0, 0, img.size[0] - self.transX, img.size[1] - self.transY))
        target = target.crop((0, 0, target.size[0] - self.transX, target.size[1] - self.transY))
        return {'image': img,'target': target}


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            target = ImageOps.expand(target, border=self.padding, fill=0)

        assert img.size == target.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, target
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), target.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return {'image': img.crop((x1, y1, x1 + tw, y1 + th)),'target': target.crop((x1, y1, x1 + tw, y1 + th))}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        assert img.size == target.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return {'image': img.crop((x1, y1, x1 + tw, y1 + th)),'target': target.crop((x1, y1, x1 + tw, y1 + th))}


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = T.ColorJitter(self.brightness, self.contrast, self.saturation,self.hue)(img)
        return {'image': img, 'target': target}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        assert img.size == target.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return {'image': img,'target': target}
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return {'image': img.resize((ow, oh), Image.BILINEAR), 'target': target.resize((ow, oh), Image.NEAREST)}
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return {'image': img.resize((ow, oh), Image.BILINEAR), 'target': target.resize((ow, oh), Image.NEAREST)}


class FreeScale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        if isinstance(size, Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        return {'image': img.resize(self.size, self.interpolation),'target': target}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,'target': target}


class RandomScaleCrop(object):
    def __init__(self, size, crop_size, fill=0):
        self.size = size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        # random scale (short edge)
        short_size = random.randint(int(self.size * 0.5), int(self.size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        target = target.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        return {'image': img,'target': target}


class FlipChannels(object):
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        img = np.array(img)[:, :, ::-1]
        return {'image': Image.fromarray(img.astype(np.uint8)), 'target': target}

####################

class Relabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        assert (isinstance(target, torch.FloatTensor) or isinstance(target, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        target[target == self.olabel] = self.nlabel
        return {'image': img, 'target': target}




