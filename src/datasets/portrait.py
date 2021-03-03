# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/9 8:56
# @Author : liumin
# @File : portrait.py

from glob2 import glob
import os

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from .transforms import build_transforms


class PortraitSegmentation(Dataset):
    """
        PortraitMatting
        http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html
    """
    def __init__(self, data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(PortraitSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = build_transforms(data_cfg.TRANSFORMS)
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
            _img = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32)
            image_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), image_id
        else:
            _img, _target = np.asarray(Image.open(self._imgs[idx]).convert('RGB'), dtype=np.float32), np.asarray(
                Image.open(self._targets[idx]), dtype=np.uint8)
            _target = self.encode_segmap(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_segmap(self, mask):
        mask_cp = mask.copy()
        # contain background, index form zero, [0, 1], pls uncomment - background in conf/dicts/portrait_dict.yml
        # mask[mask > 0] = 1

        # non background index, form zero, [0], pls uncomment - background in conf/dicts/portrait_dict.yml
        mask_nonzero = mask_cp > 0
        mask_cp[:,:] = 255
        mask_cp[mask_nonzero] = 0
        return mask_cp

    def __len__(self):
        return len(self._imgs)


