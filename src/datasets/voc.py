# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/1 15:10
# @Author : liumin
# @File : voc.py
import torch
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from CvPytorch.src.utils import palette
from .transforms import custom_transforms as ctf

"""
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
"""

def get_data_transforms(input_size):
    data_transforms = {
        'train': ctf.Compose([
            ctf.Resize(input_size),
            ctf.RandomHorizontalFlip(p=0.5),
            ctf.RandomTranslation(2),
            ctf.ToTensor(),
            # ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),

        'val': ctf.Compose([
            ctf.Resize(input_size),
            ctf.ToTensor(),
            # ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]),

        'infer': ctf.Compose([
            ctf.Resize(input_size),
            ctf.ToTensor(),
            # ctf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    }
    return data_transforms


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])
    return cmap

class VOCSegmentation(Dataset):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """
    ignore_index = 255
    VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                    [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                    [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                    [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                    [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                    [0, 64, 128]]
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = get_data_transforms(data_cfg.INPUT_SIZE)[stage]
        self.target_transform = None
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.palette = palette.get_voc_palette(self.num_classes)

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
            _img = Image.open(self._imgs[idx]).convert('RGB')
            img_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            _img, _target = Image.open(self._imgs[idx]).convert('RGB'), Image.open(self._targets[idx])
            _target = self.encode_segmap(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_segmap(self, mask):
        # This is used to convert tags
        return mask

    def __len__(self):
        return len(self._imgs)