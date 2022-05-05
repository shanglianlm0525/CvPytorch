# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/24 16:11
# @Author : liumin
# @File : mini-imagenet.py

import os
import hashlib
from multiprocessing.pool import Pool
from pathlib import Path
import cv2
from glob2 import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

"""
    mini-imageNet data
    http://image-net.org/image/ILSVRC2015/ILSVRC2015_CLS-LOC.tar.gz
    https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/train.csv 
    https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/val.csv 
    https://raw.githubusercontent.com/twitter/meta-learning-lstm/master/data/miniImagenet/test.csv
"""

class MiniImageNetClassification(Dataset):
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage
        self.is_cache = self.data_cfg.CACHE if hasattr(self.data_cfg, 'CACHE') else False

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}

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
            assert data_cfg.INDICES is not None
            for line in open(data_cfg.INDICES):
                imgpath, target = line.strip().split(' ')
                self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                self._targets.append(int(target))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if self.is_cache and self.stage != 'infer':
            self.cache = self.cache_data()

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = cv2.imread(self._imgs[idx]) # BGR
            sample = {'image': _img, 'target': None}
            return self.transform(sample)
        else:
            _img = cv2.imread(self._imgs[idx]) # BGR
            _target = self._targets[idx]
            _target = self.encode_target(_target, idx)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_target(self, _target, idx):
        return _target

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
            pbar = tqdm(pool.imap_unordered(load_image_label, zip(self._imgs, self._targets)), desc=desc, total=self.__len__())
            for img_id, _img, _target in pbar:
                _target = self.encode_target(_target)
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

def load_image_label(args):
    im_file, lb_file = args
    _img, _target = _img = cv2.imread(im_file), lb_file
    return im_file, _img, _target

def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash
