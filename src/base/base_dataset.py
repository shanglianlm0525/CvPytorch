# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/6 16:13
# @Author : liumin
# @File : base_dataset.py

import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class BaseDataset(Dataset):
    def __init__(self, root_path='', transform=None, target_transform=None, stage='train'):
        super(BaseDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self._set_files()

    def _set_files(self):
        # self.imgs = ImageFolder(os.path.join(root_path,self.stage))
        self._imgs = []
        self._labels = []
        if self.stage == 'infer':
            for root, fnames, _ in sorted(os.walk(self.root_path)):
                for fname in sorted(fnames):
                    self._imgs.extend(glob.glob(os.path.join(root, fname, '*.jpg')))
        else:
            self.cls_label = [d.name for d in os.scandir(self.root_path) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(self.root_path)):
                for fname in sorted(fnames):
                    imgs = glob.glob(os.path.join(root, fname, '*.jpg'))
                    self._imgs.extend(imgs)
                    self._labels.extend([self.cls_label.index(fname) for _ in imgs])

    def __getitem__(self, idx):
        _img, _label = Image.open(self._imgs[idx]).convert('RGB'),self._labels[idx]
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _label = self.target_transform(_label)
        return _img, _label


    def __len__(self):
        return len(self._imgs)

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Stage: {}\n".format(self.stage)
        fmt_str += "    Root_path: {}".format(self.root_path)
        return fmt_str