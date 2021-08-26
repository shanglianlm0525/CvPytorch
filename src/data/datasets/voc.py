# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/1 15:10
# @Author : liumin
# @File : voc.py
import random

import cv2
import torch
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import xml.etree.ElementTree as ET
from src.utils import palette

"""
    Pascal VOC dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
"""

class VOCDetection(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.category2id = dict(zip(self.category, range(self.num_classes)))
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.palette = palette.get_voc_palette(self.num_classes)

        self.use_difficult = False

        self._imgs = list()
        self._labels = list()
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
            for line in open(data_cfg.INDICES):
                imgpath, labelpath = line.strip().split(' ')
                self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                self._labels.append(os.path.join(data_cfg.LABELS.DET_DIR, labelpath))
            assert len(self._imgs) == len(self._labels), 'len(self._imgs) should be equals to len(self._labels)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'


    def _parse_xml(self, annopath):
        anno = ET.parse(annopath).getroot()
        boxes = []
        labels = []
        height = anno.find("size").find("height").text
        width = anno.find("size").find("width").text
        for obj in anno.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.use_difficult and difficult:
                continue
            _box = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            TO_REMOVE = 1
            box = tuple(
                map(lambda x: x - TO_REMOVE, list(map(float, box)))
            )
            boxes.append(box)

            name = obj.find("name").text.lower().strip()
            labels.append(self.category2id[name])

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        target = {}
        target["height"] = torch.tensor(int(height))
        target["width"] = torch.tensor(int(width))
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        return target

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = cv2.imread(self._imgs[idx])
            img_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            _img = cv2.imread(self._imgs[idx])
            _target = self._parse_xml(self._labels[idx])

            sample = {'image': _img, 'target': _target}
            sample = self.transform(sample)
            if self.target_transform is not None:
                return self.target_transform(sample)
            else:
                return sample

    def __len__(self):
        return len(self._imgs)

    @staticmethod
    def collate_fn(batch):
        '''list[tuple(Tensor, dict]'''
        _img_list = []
        _target_list = []
        for bch in batch:
            _img_list.append(bch['image'])
            _target_list.append(bch['target'])
        sample = {'image': _img_list, 'target': _target_list}
        return sample


class VOCSegmentation(Dataset):
    ignore_index = 255
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
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
            img_id = os.path.basename(os.path.basename(self._imgs[idx]))
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            _img, _target = Image.open(self._imgs[idx]).convert('RGB'), Image.open(self._targets[idx])
            _target = self.encode_target(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)


    def encode_target(self, mask):
        # This is used to convert tags
        return mask

    def __len__(self):
        return len(self._imgs)