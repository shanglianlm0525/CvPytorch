# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/11 15:06
# @Author : liumin
# @File : coco.py

import os
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
        self.transform = transform
        self.target_transform = target_transform
        self.resize_size = [800, 1333]

        self.num_classes = len(self.dictionary)
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR, 'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.ids = list(sorted(self.coco.imgs.keys()))

        self._filter_invalid_annotation()

        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
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
        _img = np.asarray(Image.open(os.path.join(self.data_cfg.IMG_DIR, path)).convert('RGB'), dtype=np.float32)
        # _target = dict(image_id=img_id, annotations=ann)
        _target = self.encode_map(ann, idx)
        sample = {'image': _img, 'target': _target}
        return self.transform(sample)

    def encode_map(self, ann, idx):
        img_id = self.ids[idx]
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
        sample = {'image': _img_list, 'target': _target_list}
        return sample
        # return tuple(zip(*batch))