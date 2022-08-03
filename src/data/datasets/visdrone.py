# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/17 14:23
# @Author : liumin
# @File : visdrone.py

import os
import random

import cv2
from PIL.Image import Image
from glob2 import glob
from tqdm import tqdm
import hashlib
from multiprocessing.pool import Pool
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

"""
    VisDrone Detection
    http://aiskyeye.com/
"""

class VisDroneDetection(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VisDroneDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform
        self.is_cache = self.data_cfg.CACHE if hasattr(self.data_cfg, 'CACHE') else False

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.category2id = dict(zip(self.category, range(self.num_classes)))
        self.id2category = {v: k for k, v in self.category2id.items()}

        self._imgs = []
        self._targets = []
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
            if data_cfg.INDICES is not None:
                for line in open(data_cfg.INDICES):
                    imgpath, labelpath = line.strip().split(' ')
                    self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                    self._targets.append(os.path.join(data_cfg.LABELS.DET_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, self.stage, 'images', '*', data_cfg.IMG_SUFFIX))
                self._targets = glob(
                    os.path.join(data_cfg.LABELS.DET_DIR, self.stage, 'annotations', '*', data_cfg.LABELS.DET_SUFFIX))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

        self.ids = range(len(self._imgs))
        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if self.is_cache and self.stage != 'infer':
            self.cache = self.cache_data()

    def _parse_txt(self, annopath, height, width):
        anno = pd.read_csv(annopath, header=None).values
        boxes = []
        labels = []
        for row in anno:  # x,y,w,h,score,category,truncation,occlusion
            if (row[4] == 1 and 0 < row[5] < 11):
                boxes.append([row[0], row[0], row[0] + row[2], row[1] + row[3]])
                if self.num_classes >10:
                    labels.append(row[5])
                else:
                    labels.append(row[5] - 1)

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]

        target = {}
        target["height"] = torch.tensor(int(height))
        target["width"] = torch.tensor(int(width))
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels)
        return target


    def __getitem__(self, idx):
        if self.load_num > 1:
            idxs = [idx] + random.choices(self.ids, k=self.load_num - 1)
            sample = []
            for idx in idxs:
                if self.is_cache:
                    s = self.cache[idx]
                else:
                    _img = cv2.imread(self._imgs[idx])
                    _target = self._parse_txt(self._targets[idx], _img.shape[0], _img.shape[1])
                    s = {'image': _img, 'target': _target}
                sample.append(s)
        else:
            if self.is_cache:
                sample = self.cache[idx]
            else:
                _img = cv2.imread(self._imgs[idx])
                _target = self._parse_txt(self._targets[idx], _img.shape[0], _img.shape[1])
                sample = {'image': _img, 'target': _target}

        sample = self.transform(sample)

        if self.target_transform is not None:
            return self.target_transform(sample)
        else:
            return sample

    def __len__(self):
        return len(self._imgs)

    def cache_data(self):
        cache = {}  # dict
        NUM_THREADS = 8
        cache_path = (Path(self.data_cfg.IMG_DIR)/self.stage).with_suffix('.cache')
        '''
        if cache_path.is_file():
            cache = np.load(cache_path, allow_pickle=True).item()
            if cache['hash'] == get_hash(self._imgs + self._targets):
                return cache
        '''
        desc = f"Scanning '{cache_path.parent / cache_path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(pool.imap_unordered(load_det_image_label, zip(self._imgs, self._targets)),
                desc=desc, total=self.__len__())
            for img_id, _img, _target in pbar:
                _target = self._parse_txt(_target, _img.shape[0], _img.shape[1])
                sample = {'image': _img, 'target': _target}
                cache[img_id] = sample
            pbar.close()
        cache['hash'] = get_hash(self._imgs + self._targets)
        try:
            np.save(cache_path, cache)  # save cache for next time
            cache_path.with_suffix('.cache.npy').rename(cache_path)  # remove .npy suffix
            print(f'{self.stage} :New cache created: {cache_path}')
        except Exception as e:
            print(f'{self.stage} :WARNING: Cache directory {cache_path.parent} is not writeable: {e}')  # path not writeable
        return cache

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


def load_det_image_label(args):
    im_file, lb_file = args
    _img, _target = cv2.imread(im_file), lb_file
    return im_file, _img, _target


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class VisDroneTrack(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VisDroneTrack, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.stage = stage
        self.load_num = data_cfg.LOAD_NUM if data_cfg.__contains__('LOAD_NUM') and self.stage=='train' else 1
        self.transform = transform
        self.target_transform = target_transform
        self.is_cache = self.data_cfg.CACHE if hasattr(self.data_cfg, 'CACHE') else False

        self.num_classes = len(self.dictionary)

    def __getitem__(self, idx):
        pass

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

        sample = {'image': torch.stack(_img_list, 0), 'target': _target_list}
        return sample
