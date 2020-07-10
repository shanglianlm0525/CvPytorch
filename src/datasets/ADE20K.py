# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/6 16:00
# @Author : liumin
# @File : ADE20K.py
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset

class ADE20KDataset():
    """
        ADE20K dataset
        http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, root_path='/home/lmin/data/ADE20K/images/training', transform=None, target_transform=None, stage='train'):
        super(ADE20KDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage

        self.num_classes = 150
        # self.palette = palette.ADE20K_palette

        self._imgs = []
        self._labels = []
        if self.stage == 'infer':
            for root, fnames, _ in sorted(os.walk(self.root_path)):
                for fname in sorted(fnames):
                    self._imgs.extend(glob(os.path.join(root, fname, '*.jpg')))
        else:
            self.cls_label = [d.name for d in os.scandir(self.root_path) if d.is_dir()]
            for root, fnames, _ in sorted(os.walk(self.root_path)):
                for fname in sorted(fnames):
                    imgs = glob(os.path.join(root, fname, '*.jpg'))
                    self._imgs.extend(imgs)
                    self._labels.extend([self.cls_label.index(fname) for _ in imgs])

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/ADE20K/images/training'
    dataset = ADE20KDataset(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(20))
    print('finished!')