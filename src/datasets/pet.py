# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/11 14:25
# @Author : liumin
# @File : pet.py
import glob
import os

import torch
import torchvision.transforms as transformsT
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class PetDataset(Dataset):
    """
        The Oxford-IIIT Pet Dataset
        http://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(PetDataset, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
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
        img, label = Image.open(self._imgs[idx]).convert('RGB'),self._labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.stage=='infer':
            return img
        else:
            if self.target_transform is not None:
                label = self.target_transform(label)
            return img, label

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/Pet/train'
    dataset = PetDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(0))
    print(len(dataset.cls_label))
    print('finished!')