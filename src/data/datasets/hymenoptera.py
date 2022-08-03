# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/29 9:38
# @Author : liumin
# @File : hymenoptera.py


import os

import cv2
from glob2 import glob
import torch
from torch.utils.data import Dataset
import numpy as np

"""
    hymenoptera_data
    https://download.pytorch.org/tutorial/hymenoptera_data.zip
"""

# @accelerate_dataset()
class HymenopteraClassification(Dataset):
    def __init__(self,data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(HymenopteraClassification, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}

        self._imgs = []
        self._targets = []
        if self.stage == 'infer':
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    self._imgs.extend(glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX)))
        else:
            self.cls_label = [d.name for d in os.scandir(data_cfg.IMG_DIR) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    imgs = glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX))
                    self._imgs.extend(imgs)
                    self._targets.extend([self.cls_label.index(fname) for _ in imgs])

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            # _img = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32)
            _img = cv2.imread(self._imgs[idx])  # BGR
            sample = {'image': _img, 'target': None}
            return self.transform(sample)
        else:
            # _img, _target = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32), self._targets[idx]
            _img = cv2.imread(self._imgs[idx]) # BGR
            _target = self._targets[idx]
            _target = self.encode_target(_target, idx)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_target(self, _target, idx):
        return _target

    def __len__(self):
        return len(self._imgs)
