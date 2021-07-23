# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/6 10:24
# @Author : liumin
# @File : cityscapes2.py
import os

import cv2
import torch
from PIL import Image
from glob2 import glob
import numpy as np
import random

from PIL import Image, ImageOps, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as transformsT
from src.utils import palette

## lednet

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

# Augmentations - different function implemented to perform random augments on both image and target
class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc = enc
        self.augment = augment
        self.height = height
        pass

    def __call__(self, input, target):
        # do something to both images
        input = transformsT.Resize(self.height, Image.BILINEAR)(input)
        target = transformsT.Resize(self.height, Image.NEAREST)(target)

        if (self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0), fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0] - transX, input.size[1] - transY))
            target = target.crop((0, 0, target.size[0] - transX, target.size[1] - transY))

        input = transformsT.ToTensor()(input)
        if (self.enc):
            target = transformsT.Resize(int(self.height / 8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CityscapesSegmentation(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = None # MyCoTransform(enc=True, augment=True, height=512)
        self.target_transform = None # MyCoTransform(enc=True, augment=True, height=512)
        self.co_transform = MyCoTransform(enc=True, augment=True if stage=='train' else False, height=512)
        self.stage = stage

        self.num_classes = len(self.dictionary)+1
        self.palette = palette.CityScpates_palette

        self._imgs = list()
        self._labels = list()
        if self.stage == 'infer':
            if data_cfg.INDICES is not None:
                with open(data_cfg.INDICES, 'r') as fd:
                    self._imgs.extend([os.path.join(data_cfg.IMG_DIR, line.strip()) for line in fd])
            else:
                for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                    for fname in sorted(fnames):
                        self._imgs.extend(glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX)))

            if len(self._imgs) == 0:
                raise RuntimeError(
                    "Found 0 images in subfolders of: " + data_cfg.IMG_DIR if data_cfg.INDICES is not None else data_cfg.INDICES + "\n")
        else:
            if data_cfg.INDICES is not None:
                for line in open(data_cfg.INDICES):
                    imgpath, labelpath = line.strip().split(' ')
                    self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                    self._labels.append(os.path.join(data_cfg.LABELS.SEG_DIR, labelpath))

            assert len(self._imgs) == len(self._labels), 'len(self._imgs) should be equals to len(self._labels)'


    def __getitem__(self, idx):
        img = Image.open(self._imgs[idx]).convert('RGB')
        if self.stage == 'infer':
            if self.transform is not None:
                img = self.transform(img)
            image_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            return img, image_id

        mask = Image.open(self._labels[idx]).convert('P')

        if self.co_transform is not None:
            img, mask = self.co_transform(img, mask)

        return img, mask

    def __len__(self):
        return len(self._imgs)
