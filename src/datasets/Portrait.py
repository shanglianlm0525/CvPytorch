# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/9 8:56
# @Author : liumin
# @File : Portrait.py

import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class PortraitDataset(Dataset):
    """
        PortraitMatting
        http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html
    """
    def __init__(self, data_cfg, transform=None,target_transform=None, stage='train'):
        super(PortraitDataset, self).__init__()
        self.data_cfg = data_cfg
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self._imgs = []
        self._labels = []
        if self.stage == 'infer':
            self._imgs = glob.glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
        else:
            self._imgs = glob.glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
            self._labels = glob.glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))

    def __getitem__(self, idx):
        _img, _label = Image.open(self._imgs[idx]).convert('RGB'),Image.open(self._labels[idx]).convert("L")
        if self.transform is not None:
            _img = self.transform(_img)
        if self.stage=='infer':
            return _img
        else:
            if self.target_transform is not None:
                _label = self.target_transform(_label)
            return _img, _label

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/portrait/train/images'
    dataset = PortraitDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(0))
    print(len(dataset))
    print('finished!')