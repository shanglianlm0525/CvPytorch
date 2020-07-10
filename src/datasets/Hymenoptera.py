# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/29 9:38
# @Author : liumin
# @File : Hymenoptera.py

import glob
import os

import torch
import torchvision.transforms as transformsT
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class HymenopteraDataset(Dataset):
    """
        hymenoptera_data
        https://download.pytorch.org/tutorial/hymenoptera_data.zip
    """
    def __init__(self,data_cfg, transform=None,target_transform=None, stage='train'):
        super(HymenopteraDataset, self).__init__()
        self.data_cfg = data_cfg
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self._imgs = []
        self._labels = []
        # self.imgs = ImageFolder(root_path)
        if self.stage == 'infer':
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    self._imgs.extend(glob.glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX)))
        else:
            self.cls_label = [d.name for d in os.scandir(data_cfg.IMG_DIR) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(data_cfg.IMG_DIR)):
                for fname in sorted(fnames):
                    imgs = glob.glob(os.path.join(root, fname, data_cfg.IMG_SUFFIX))
                    self._imgs.extend(imgs)
                    self._labels.extend([self.cls_label.index(fname) for _ in imgs])


    def __getitem__(self, idx):
        _img, _label = Image.open(self._imgs[idx]).convert('RGB'),self._labels[idx]
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
    root_path = '/home/lmin/data/hymenoptera/train'
    dataset = HymenopteraDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(0))
    print(len(dataset.cls_label))
    print('finished!')