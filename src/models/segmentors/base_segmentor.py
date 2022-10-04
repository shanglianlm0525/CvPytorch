# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/2 18:05
# @Author : liumin
# @File : base_segmentor.py

import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import torch.nn as nn


class BaseSegmentor(nn.Module, metaclass=ABCMeta):
    """Base class for seg."""

    def __init__(self):
        super(BaseSegmentor, self).__init__()


    @property
    def with_backbone(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.backbone is not None

    @property
    def with_neck(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'head') and self.head is not None

    @property
    def with_auxiliary_head(self):
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self,'auxiliary_head') and self.auxiliary_head is not None

    @property
    def with_loss(self):
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'loss') and self.loss is not None

    @abstractmethod
    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        pass