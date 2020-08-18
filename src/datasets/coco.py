# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:30
# @Author : liumin
# @File : coco.py
import cv2
import torch
from glob2 import glob
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.utils import palette
from pycocotools.coco import COCO
import numpy as np

from CvPytorch.src.models.ext.ssd.augmentations import SSDAugmentation


def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self,coco_root):
        self.label_map = get_label_map(os.path.join(coco_root, 'coco_labels.txt'))

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]
                label_idx = self.label_map[obj['category_id']] - 1
                final_box = list(np.array(bbox)/scale)
                final_box.append(label_idx)
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("no bbox problem!")

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]

class CocoDetection(Dataset):
    """
        MS Coco Detection
        http://mscoco.org/dataset/#detections-challenge2016
    """
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(CocoDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = SSDAugmentation()
        self.target_transform = COCOAnnotationTransform(coco_root=os.path.dirname(data_cfg.LABELS.DET_DIR))
        self.stage = stage

        self.num_classes = 80
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR,'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.ids = list(self.coco.imgToAnns.keys())

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        target = self.coco.imgToAnns[img_id]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)

        target = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_cfg.IMG_DIR, self.coco.loadImgs(img_id)[0]['file_name'])
        assert os.path.exists(path), 'Image path does not exist: {}'.format(path)
        img = cv2.imread(os.path.join(self.data_cfg.IMG_DIR, path))
        height, width, _ = img.shape
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)
        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target

    def __len__(self):
        return len(self.ids)

