# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/9 8:56
# @Author : liumin
# @File : portrait.py
import cv2
from glob2 import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as tf
from .transforms import custom_transforms as ctf


data_transforms = {
    'train': tf.Compose([
        ctf.Resize((600,800)),
        ctf.RandomHorizontalFlip(p=0.5),
        ctf.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
        ctf.RandomRotate(5),
        # ctf.RandomTranslation(2),
        ctf.ToTensor(),
        ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),

    'val': tf.Compose([
        ctf.Resize((600, 800)),
        ctf.ToTensor(),
        ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]),

    'infer': tf.Compose([
        ctf.Resize((600, 800)),
        ctf.ToTensor(),
        ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
}


class PortraitSegmentation(Dataset):
    """
        PortraitMatting
        http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html
    """
    def __init__(self, data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(PortraitSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = data_transforms[stage]
        self.target_transform = None
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}

        self._imgs = []
        self._targets = []
        if self.stage == 'infer':
            if data_cfg.INDICES is not None:
                [self._imgs.append(os.path.join(data_cfg.IMG_DIR, line.strip())) for line in open(data_cfg.INDICES)]
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
        else:
            if data_cfg.INDICES is not None:
                for line in open(data_cfg.INDICES):
                    imgpath, labelpath = line.strip().split(' ')
                    self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                    self._targets.append(os.path.join(data_cfg.LABELS.DET_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
                self._targets = glob(os.path.join(data_cfg.LABELS.SEG_DIR, data_cfg.LABELS.SEG_SUFFIX))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = Image.open(self._imgs[idx]).convert('RGB')
            image_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), image_id
        else:
            _img, _target = Image.open(self._imgs[idx]).convert('RGB'), Image.open(self._targets[idx])
            _target = self.encode_segmap(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_segmap(self, mask):
        mask = np.array(mask, dtype=np.uint8)
        # contain background, index form zero, [0, 1], pls uncomment - background in conf/dicts/portrait_dict.yml
        # mask[mask > 0] = 1

        # non background index, form zero, [0], pls uncomment - background in conf/dicts/portrait_dict.yml
        mask_nonzero = mask > 0
        mask[:,:] = 255
        mask[mask_nonzero] = 0

        mask = Image.fromarray(mask)
        return mask

    def __len__(self):
        return len(self._imgs)


