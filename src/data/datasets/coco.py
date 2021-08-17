# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/11 15:06
# @Author : liumin
# @File : coco.py

import os
import random

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np

"""
    MS Coco Detection
    http://mscoco.org/dataset/#detections-challenge2016
"""


class CocoDetection(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(CocoDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform

        self.num_classes = len(self.dictionary)
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR, 'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.ids = list(sorted(self.coco.imgs.keys()))

        self._filter_invalid_annotation()

        if self.num_classes > 80: # BACKGROUND_AS_CATEGORY
            self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        else:
            self.category2id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def _filter_invalid_annotation(self):
        # check annos, filtering invalid data
        valid_ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                valid_ids.append(id)
        self.ids = valid_ids

    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot):
            return False
        return True

    def __getitem__(self, idx):
        if self.load_num > 1:
            img_ids = [self.ids[idx]] + random.choices(self.ids, k=self.load_num - 1)
            sample = []
            for img_id in img_ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                ann = self.coco.loadAnns(ann_ids)

                path = self.coco.loadImgs(img_id)[0]['file_name']
                assert os.path.exists(os.path.join(self.data_cfg.IMG_DIR, path)), 'Image path does not exist: {}'.format(
                    os.path.join(self.data_cfg.IMG_DIR, path))

                _img = cv2.imread(os.path.join(self.data_cfg.IMG_DIR, path))
                _target = self.encode_map(ann, img_id)
                s = {'image': _img, 'target': _target}
                sample.append(s)
        else:
            img_id = self.ids[idx]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            ann = self.coco.loadAnns(ann_ids)

            path = self.coco.loadImgs(img_id)[0]['file_name']
            assert os.path.exists(os.path.join(self.data_cfg.IMG_DIR, path)), 'Image path does not exist: {}'.format(
                os.path.join(self.data_cfg.IMG_DIR, path))

            _img = cv2.imread(os.path.join(self.data_cfg.IMG_DIR, path))
            _target = self.encode_map(ann, img_id)
            sample = {'image': _img, 'target': _target}

        sample = self.transform(sample)

        if self.target_transform is not None:
            return self.target_transform(sample)
        else:
            return sample

    def encode_map(self, ann, img_id):
        target = dict(image_id=img_id, annotations=ann)
        return target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        '''list[tuple(Tensor, dict]'''
        _img_list = []
        _target_list = []
        for bch in batch:
            _img_list.append(bch['image'])
            _target_list.append(bch['target'])

        sample = {'image': torch.stack(_img_list, 0), 'target': _target_list}
        return sample


class CocoSegmentation(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(CocoSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform

        self.num_classes = len(self.dictionary)
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR, 'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.ids = list(sorted(self.coco.imgs.keys()))

        self._filter_invalid_annotation()

        ## need to process
        self.category2id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

    def _filter_invalid_annotation(self):
        # check annos, filtering invalid data
        valid_ids = []
        for id in self.ids:
            ann_id = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                valid_ids.append(id)
        self.ids = valid_ids

    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot):
            return False
        return True

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        assert os.path.exists(os.path.join(self.data_cfg.IMG_DIR, path)), 'Image path does not exist: {}'.format(
            os.path.join(self.data_cfg.IMG_DIR, path))

        _img = Image.open(os.path.join(self.data_cfg.IMG_DIR, path)).convert('RGB')
        _target = self.encode_map(ann, idx)
        sample = {'image': _img, 'target': _target}
        return self.transform(sample)

    def encode_map(self, ann, idx):
        img_id = self.ids[idx]
        target = dict(image_id=img_id, annotations=ann)
        return target

    def __len__(self):
        return len(self.ids)