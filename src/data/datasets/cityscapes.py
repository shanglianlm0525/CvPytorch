# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/6 10:24
# @Author : liumin
# @File : cityscapes.py

import os
import hashlib
from multiprocessing.pool import Pool
from pathlib import Path

from glob2 import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import palette

"""
    Cityscapes dataset
    https://www.cityscapes-dataset.com/
"""

class CityscapesSegmentation(Dataset):
    ignore_index = 255
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage
        self.is_cache = self.data_cfg.CACHE if hasattr(self.data_cfg, 'CACHE') else False

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.category2id = dict(zip(self.category, range(self.num_classes)))
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.palette = palette.CityScpates_palette

        self.invalid_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

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
                    self._targets.append(os.path.join(data_cfg.LABELS.SEG_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, 'leftImg8bit',self.stage,'*',data_cfg.IMG_SUFFIX))
                self._targets = glob(os.path.join(data_cfg.LABELS.SEG_DIR,'gtFine',self.stage,'*', data_cfg.LABELS.SEG_SUFFIX))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        if self.is_cache and self.stage != 'infer':
            self.cache = self.cache_data()

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = Image.open(self._imgs[idx]).convert('RGB')
            img_id = os.path.basename(os.path.basename(self._imgs[idx]))
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            if self.is_cache:
                sample = self.cache[self._imgs[idx]]
            else:
                _img, _target = Image.open(self._imgs[idx]).convert('RGB'), Image.open(self._targets[idx])
                _target = self.encode_target(_target)
                sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_target(self, target):
        # This is used to convert tags
        target = np.asarray(target, dtype=np.uint8).copy()
        # Put all void classes to zero
        for _voidc in self.invalid_classes:
            target[target == _voidc] = self.ignore_index
        # index from zero 0:18
        for _validc in self.valid_classes:
            target[target == _validc] = self.class_map[_validc]
        target = Image.fromarray(target.astype(np.uint8))
        return target

    @classmethod
    def decode_target(self, target):
        target[target == 255] = 19
        return target

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
            pbar = tqdm(pool.imap_unordered(load_image_label, zip(self._imgs, self._targets)),
                desc=desc, total=self.__len__())
            for im_file, _img, _target in pbar:
                _target = self.encode_target(_target)
                sample = {'image': _img, 'target': _target}
                cache[im_file] = sample
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
    _img, _target = Image.open(im_file).convert('RGB'), Image.open(lb_file)
    return im_file, _img, _target


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


class CityscapesSegmentationBAK(Dataset):
    ignore_index = 255
    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = transform
        self.target_transform = target_transform
        self.stage = stage
        self.cache_image = self.data_cfg.CACHE_IMAGE if hasattr(self.data_cfg, 'CACHE_IMAGE') else False
        self.cache_label = self.data_cfg.CACHE_LABEL if hasattr(self.data_cfg, 'CACHE_LABEL') else False

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.category2id = dict(zip(self.category, range(self.num_classes)))
        self.id2category = {v: k for k, v in self.category2id.items()}
        self.palette = palette.CityScpates_palette

        self.invalid_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_classes, range(self.num_classes)))

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
                    self._targets.append(os.path.join(data_cfg.LABELS.SEG_DIR, labelpath))
            else:
                self._imgs = glob(os.path.join(data_cfg.IMG_DIR, 'leftImg8bit',self.stage,'*',data_cfg.IMG_SUFFIX))
                self._targets = glob(os.path.join(data_cfg.LABELS.SEG_DIR,'gtFine',self.stage,'*', data_cfg.LABELS.SEG_SUFFIX))

            assert len(self._imgs) == len(self._targets), 'len(self._imgs) should be equals to len(self._targets)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

        self.caches()

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

    def encode_target(self, target):
        # This is used to convert tags
        target = np.asarray(target, dtype=np.uint8).copy()
        # Put all void classes to zero
        for _voidc in self.invalid_classes:
            target[target == _voidc] = self.ignore_index
        # index from zero 0:18
        for _validc in self.valid_classes:
            target[target == _validc] = self.class_map[_validc]
        target = Image.fromarray(target.astype(np.uint8))
        return target

    @staticmethod
    def decode_target(self, target):
        target[target == 255] = 19
        return target

    def __len__(self):
        return len(self._imgs)