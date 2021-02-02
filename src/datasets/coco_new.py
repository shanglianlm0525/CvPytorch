# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/8 15:58
# @Author : liumin
# @File : coco_new.py

import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch._six import container_abcs, string_classes, int_classes
from torch.utils.data import Dataset

from CvPytorch.src.datasets.nanodet_transform import Pipeline

"""
    MS Coco Detection
    http://mscoco.org/dataset/#detections-challenge2016
"""

class CocoDetection(Dataset):
    """
          A base class of detection dataset. Referring from MMDetection.
          A dataset should have images, annotations and preprocessing pipelines
          NanoDet use [xmin, ymin, xmax, ymax] format for box and
           [[x0,y0], [x1,y1] ... [xn,yn]] format for key points.
          instance masks should decode into binary masks for each instance like
          {
              'bbox': [xmin,ymin,xmax,ymax],
              'mask': mask
           }
          segmentation mask should decode into binary masks for each class.

          :param img_path: image data folder
          :param ann_path: annotation file path or folder
          :param use_instance_mask: load instance segmentation data
          :param use_seg_mask: load semantic segmentation data
          :param use_keypoint: load pose keypoint data
          :param load_mosaic: using mosaic data augmentation from yolov4
          :param mode: train or val or test
    """
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(CocoDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.num_classes = len(self.dictionary)

        self.img_path = data_cfg.IMG_DIR
        self.ann_path = os.path.join(data_cfg.LABELS.DET_DIR,'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR)))

        self.input_size = [320, 320]
        if self.stage == 'train':
            pipeline = {'perspective': 0.0,
                        'scale': [0.6, 1.4],
                        'stretch': [[1, 1], [1, 1]],
                        'rotation': 0,
                        'shear': 0,
                        'translate': 0,
                        'flip': 0.5,
                        'brightness': 0.2,
                        'contrast': [0.8, 1.2],
                        'saturation': [0.8, 1.2],
                        'normalize': [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}
        else:
            pipeline = {'normalize': [[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]}
        self.pipeline = Pipeline(pipeline, keep_ratio=True)

        self.data_info = self.get_data_info(self.ann_path)

        self.use_instance_mask = False
        self.use_seg_mask = False
        self.use_keypoint = False

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if self.stage == 'infer':
            return self.get_val_data(idx)
        else:
            meta = self.get_train_data(idx)
            sample = {'image': meta['img'], 'target': meta}
            return sample
