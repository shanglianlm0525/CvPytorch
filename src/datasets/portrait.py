# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/9 8:56
# @Author : liumin
# @File : portrait.py
import cv2
from glob2 import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms as transformsT

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(image).float().unsqueeze(0)

def transforms_portrait_img():
    data_transforms = {
        'train': transformsT.Compose([
            transformsT.Resize((800,600)),
            # transformsT.RandomHorizontalFlip(),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'val': transformsT.Compose([
            transformsT.Resize((800,600)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'infer': transformsT.Compose([
            transformsT.Resize((800,600)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def transforms_portrait_seg():
    data_transforms = {
        'train': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ]),

        'val': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ]),

        'infer': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ])
    }
    return data_transforms


class PortraitSegmentation(Dataset):
    """
        PortraitMatting
        http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html
    """
    def __init__(self, data_cfg, dictionary=None, transform=None,target_transform=None, stage='train'):
        super(PortraitSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transforms_portrait_img()[stage]
        self.target_transform = transforms_portrait_seg()[stage]
        self.stage = stage

        self._imgs = []
        self._labels = []
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
                    self._labels.append(os.path.join(data_cfg.LABELS.DET_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, data_cfg.IMG_SUFFIX))
                self._labels = glob(os.path.join(data_cfg.LABELS.SEG_DIR, data_cfg.LABELS.SEG_SUFFIX))

            assert len(self._imgs) == len(self._labels), 'len(self._imgs) should be equals to len(self._labels)'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            img = Image.open(self._imgs[idx]).convert('RGB')
        else:
            img, label = Image.open(self._imgs[idx]).convert('RGB'),cv2.imread(self._labels[idx],0)
        if self.transform is not None:
            img = self.transform(img)
        if self.stage=='infer':
            image_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            return img,image_id
        else:
            label = label /255
            label = cv2.resize(label,(600,800))
            if self.target_transform is not None:
                label = self.target_transform(label)
            return img, label

    def __len__(self):
        return len(self._imgs)


