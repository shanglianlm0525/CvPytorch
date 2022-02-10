# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/11 15:06
# @Author : liumin
# @File : coco.py
import hashlib
import os
import random
from itertools import repeat
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm

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
        self.is_cache = self.data_cfg.CACHE if hasattr(self.data_cfg, 'CACHE') else False

        self.num_classes = len(self.dictionary)
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR, 'instances_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.ids = list(sorted(self.coco.getImgIds()))

        self._filter_invalid_annotation()

        self.category2id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        # just for FCOS
        if hasattr(data_cfg.TRANSFORMS, 'FilterAndRemapCocoCategories') \
                and hasattr(data_cfg.TRANSFORMS.FilterAndRemapCocoCategories, 'categories') \
                and len(data_cfg.TRANSFORMS.FilterAndRemapCocoCategories.categories) > 80:
            self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if self.is_cache and self.stage != 'infer':
            self.imgpaths = []
            self.anns = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                ann = self.coco.loadAnns(ann_ids)
                path = self.coco.loadImgs(img_id)[0]['file_name']
                self.imgpaths.append(os.path.join(self.data_cfg.IMG_DIR, path))
                self.anns.append(ann)
            self.cache = self.cache_data()

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
                if self.is_cache:
                    s = self.cache[self.ids[idx]]
                else:
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
            if self.is_cache:
                sample = self.cache[self.ids[idx]]
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

    def cache_data(self):
        cache = {}  # dict
        NUM_THREADS = 8
        cache_path = (Path(self.data_cfg.IMG_DIR).parent.parent/self.stage).with_suffix('.cache')
        '''
        if cache_path.is_file():
            cache = np.load(cache_path, allow_pickle=True).item()
            if cache['hash'] == get_hash(self._imgs + self._targets):
                return cache
        '''
        desc = f"Scanning '{cache_path.parent / cache_path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(load_image_label, zip(self.ids, self.imgpaths, self.anns)), desc=desc, total=self.__len__())
            for img_id, _img, ann in pbar:
                _target = self.encode_map(ann, img_id)
                sample = {'image': _img, 'target': _target}
                cache[img_id] = sample
            pbar.close()
        cache['hash'] = get_hash(self.imgpaths)
        try:
            np.save(cache_path, cache)  # save cache for next time
            cache_path.with_suffix('.cache.npy').rename(cache_path)  # remove .npy suffix
            print(f'{self.stage} :New cache created: {cache_path}')
        except Exception as e:
            print(f'{self.stage} :WARNING: Cache directory {cache_path.parent} is not writeable: {e}')  # path not writeable
        return cache


def load_image_label(args):
    img_id, imgpath, ann = args
    _img = cv2.imread(imgpath)
    return img_id, _img, ann


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class CocoKeypoint(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(CocoKeypoint, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform

        self.num_classes = len(self.dictionary)
        self.coco = COCO(os.path.join(data_cfg.LABELS.DET_DIR, 'person_keypoints_{}.json'.format(os.path.basename(data_cfg.IMG_DIR))))
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.ids = list(sorted(self.coco.getImgIds(catIds=self.cat_ids)))

        self._filter_for_keypoint_annotations()

    def _filter_for_keypoint_annotations(self):
        def has_keypoint_annotation(image_id):
            ann_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.cat_ids)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if 'keypoints' not in ann:
                    continue
                if any(v > 0.0 for v in ann['keypoints'][2::3]):
                    return True
            return False

        self.ids = [image_id for image_id in self.ids if has_keypoint_annotation(image_id)]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids)
        ann = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        assert os.path.exists(os.path.join(self.data_cfg.IMG_DIR, path)), 'Image path does not exist: {}'.format(
            os.path.join(self.data_cfg.IMG_DIR, path))

        # _img = Image.open(os.path.join(self.data_cfg.IMG_DIR, path)).convert('RGB')
        _img = cv2.imread(os.path.join(self.data_cfg.IMG_DIR, path))
        _target = self.encode_map(ann, idx)
        sample = {'image': _img, 'target': _target}
        sample =  self.transform(sample)

        if self.target_transform is not None:
            return self.target_transform(sample)
        else:
            return sample

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
        self.ids = list(sorted(self.coco.getImgIds()))

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

        # _img = Image.open(os.path.join(self.data_cfg.IMG_DIR, path)).convert('RGB')
        _img = cv2.imread(os.path.join(self.data_cfg.IMG_DIR, path))
        _target = self.encode_map(ann, idx)
        sample = {'image': _img, 'target': _target}
        return self.transform(sample)

    def encode_map(self, ann, idx):
        img_id = self.ids[idx]
        target = dict(image_id=img_id, annotations=ann)
        return target

    def __len__(self):
        return len(self.ids)