# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/11 13:30
# @Author : liumin
# @File : flower.py
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class FlowerDataset(Dataset):
    """
        17 Category Flower Dataset
        http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
        102 Category Flower Dataset
        http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html
    """
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(FlowerDataset, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        # self.imgs = ImageFolder(os.path.join(root_path,self.stage))
        self._imgs = []
        self._labels = []
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
                    self._labels.extend([self.cls_label.index(fname) for _ in imgs])

    def __getitem__(self, idx):
        img, label = Image.open(self._imgs[idx]).convert('RGB'),self._labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label


    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/Flower17/train'
    dataset = FlowerDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(20))
    print('finished!')
