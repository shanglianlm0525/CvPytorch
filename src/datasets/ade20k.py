# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/6 16:00
# @Author : liumin
# @File : ade20k.py
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class ADE20KDataset(Dataset):
    """
        ADE20K dataset
        http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(ADE20KDataset, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self.num_classes = 150
        # self.palette = palette.ADE20K_palette

        self._imgs = []
        self._labels = []
        if self.stage == 'infer':
            self._imgs = glob.glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
        else:
            self._imgs = glob.glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
            self._labels = glob.glob(os.path.join(data_cfg.LABELS.SEG_DIR, data_cfg.LABELS.SEG_SUFFIX))

    def __getitem__(self, idx):
        img, label = Image.open(self._imgs[idx]).convert('RGB'), np.asarray(Image.open(self._labels[idx]), dtype=np.int32) - 1 # -1:149
        if self.transform is not None:
            img = self.transform(img)
        if self.stage == 'infer':
            return img
        else:
            if self.target_transform is not None:
                label = self.target_transform(label)
            return img, label

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/ADE20K/images/training'
    dataset = ADE20KDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(20))
    print('finished!')