# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/23 11:02
# @Author : liumin
# @File : voc_det.py
import torch
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import xml.etree.ElementTree as ET
from ..utils import palette

class VOCDetection(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCDetection, self).__init__()
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

        self.use_difficult = False

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
            _img = Image.open(self._imgs[idx]).convert('RGB')
            _target = self.parse_annotation(self._targets[idx], idx)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def parse_annotation(self, annopath, idx):
        anno = ET.parse(annopath).getroot()
        boxes = []
        classes = []
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
            classes.append(self.name2id[name])

        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        return target


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
        # return tuple(zip(*batch))