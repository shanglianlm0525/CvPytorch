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
from CvPytorch.src.utils import palette
from CvPytorch.src.datasets.transforms import build_transforms

"""
    Cityscapes dataset
    https://www.cityscapes-dataset.com/
"""

class ADE20KSegmentation(Dataset):
    ignore_index = 255
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(ADE20KSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = build_transforms(data_cfg.TRANSFORMS)
        self.target_transform = target_transform
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.palette = palette.ADE20K_palette

        self._imgs = []
        self._targets = []
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
                    self._targets.append(os.path.join(data_cfg.LABELS.SEG_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR,data_cfg.IMG_SUFFIX))
                self._targets = glob(os.path.join(data_cfg.LABELS.SEG_DIR, data_cfg.LABELS.SEG_SUFFIX))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32)
            img_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            _img, _target = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32), np.asarray(
                Image.open(self._targets[idx]), dtype=np.uint8)
            _target = self.encode_segmap(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_segmap(self, mask):
        # This is used to convert tags
        # index from zero 0:149
        mask_cp = mask.copy()
        mask_cp = mask_cp -1 # uint8 will change -1 to 255
        return mask_cp

    def __len__(self):
        return len(self._imgs)


if __name__ == '__main__':
    root_path = '/home/lmin/data/ADE20K/images/training'
    dataset = ADE20KSegmentation(root_path)

    print(dataset.__len__())
    print(dataset.__getitem__(20))
    print('finished!')