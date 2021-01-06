# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/11 11:19
# @Author : liumin
# @File : nanodet_coco.py

import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch._six import container_abcs, string_classes, int_classes

from .nanodet_transform.pipeline import Pipeline
from torch.utils.data import Dataset

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
        self.ann_path = os.path.join(data_cfg.LABELS.DET_DIR, 'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR)))

        self.input_size = [320, 320]
        if self.stage=='train':
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

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)

        # check annos, filtering invalid data
        valid_img_ids = []
        for id in self.img_ids:
            ann_id = self.coco_api.getAnnIds(imgIds=id, iscrowd=None)
            ann = self.coco_api.loadAnns(ann_id)
            if self._has_valid_annotation(ann):
                valid_img_ids.append(id)
        self.img_ids = valid_img_ids

        return img_info

    def _has_only_empty_bbox(self,annot):
        return all(any(o <= 1 for o in obj['bbox'][2:]) for obj in annot)


    def _has_valid_annotation(self,annot):
        if len(annot) == 0:
            return False

        if self._has_only_empty_bbox(annot):
            return False

        return True

    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])
        anns = self.coco_api.loadAnns(ann_ids)
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        if self.use_instance_mask:
            annotation['masks'] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)
        img = cv2.imread(image_path)
        ann = self.get_img_annotation(idx)
        meta = dict(img=img, img_info=img_info, gt_bboxes=ann['bboxes'], gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)

    @staticmethod
    def collate_fn(batch):
        return collate_function(batch)


def collate_function(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        # TODO: support pytorch < 1.3
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.storage()._new_shared(numel)
        #     out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            # return collate_function([torch.as_tensor(b) for b in batch])
            return batch
        elif elem.shape == ():  # scalars
            # return torch.as_tensor(batch)
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: collate_function([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_function(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        transposed = zip(*batch)
        return [collate_function(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))