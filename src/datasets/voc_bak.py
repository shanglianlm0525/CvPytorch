# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 14:15
# @Author : liumin
# @File : voc_old.py


import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from glob2 import glob
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET

from torchvision import transforms as tf
from .transforms import custom_transforms as ctf

from src.utils import palette
from src.models.ext.ssd.augmentations import SSDAugmentation


VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(Dataset):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCDetection, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = SSDAugmentation()
        self.target_transform = VOCAnnotationTransform()
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.palette = palette.get_voc_palette(self.num_classes)

        self._imgs = list()
        self._labels = list()
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
            for line in open(data_cfg.INDICES):
                imgpath,labelpath = line.strip().split(' ')
                self._imgs.append(os.path.join(data_cfg.IMG_DIR, imgpath))
                self._labels.append(os.path.join(data_cfg.LABELS.DET_DIR, labelpath))
            assert len(self._imgs) == len(self._labels), 'len(self._imgs) should be equals to len(self._labels)'
            assert len(self._imgs) > 0, 'Found 0 images in the specified location, pls check it!'

    def __getitem__(self, idx):
        if self.stage == 'infer':
            _img = cv2.imread(self._imgs[idx])
            img_id = os.path.splitext(os.path.basename(self._imgs[idx]))[0]
            sample = {'image': _img, 'mask': None}
            return self.transform(sample), img_id
        else:
            img = cv2.imread(self._imgs[idx])
            target = ET.parse(self._labels[idx]).getroot()

            height, width, channels = img.shape

            if self.target_transform is not None:
                target = self.target_transform(target, width, height)

            if self.transform is not None:
                target = np.array(target)
                img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
                # to rgb
                img = img[:, :, (2, 1, 0)]
                # img = img.transpose(2, 0, 1)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            sample = {'image': torch.from_numpy(img).permute(2, 0, 1), 'target': target}
            return sample


    def __len__(self):
        return len(self._imgs)

    @staticmethod
    def collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).
        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations
        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on
                                     0 dim
        """
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
        return torch.stack(imgs, 0), targets


data_transforms = {
    'train': tf.Compose([
        ctf.RandomHorizontalFlip(p=0.5),
        ctf.RandomScaleCrop(512,512),
        ctf.RandomGaussianBlur(p=0.2),
        ctf.ToTensor(),
        ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),

    'val': tf.Compose([
        ctf.FixScaleCrop(512),
        ctf.ToTensor(),
        ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ]),

    'infer': tf.Compose([
        ctf.FixScaleCrop(512),
        ctf.ToTensor(),
        ctf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
}

class VOCSegmentation(Dataset):
    """
    Pascal Voc dataset
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    """

    def __init__(self, data_cfg, dictionary=None, transform=None, target_transform=None, stage='train'):
        super(VOCSegmentation, self).__init__()
        self.data_cfg = data_cfg
        self.dictionary = dictionary
        self.transform = data_transforms[stage]
        self.target_transform = None
        self.stage = stage

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.name2id = dict(zip(self.category, range(self.num_classes)))
        self.id2name = {v: k for k, v in self.name2id.items()}
        self.palette = palette.get_voc_palette(self.num_classes)

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
            _img, _target = Image.open(self._imgs[idx]).convert('RGB'), Image.open(self._targets[idx])
            _target = self.encode_segmap(_target)
            sample = {'image': _img, 'target': _target}
            return self.transform(sample)

    def encode_segmap(self, mask):
        return mask

    def __len__(self):
        return len(self._imgs)

