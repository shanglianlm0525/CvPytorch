# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/2 18:05
# @Author : liumin
# @File : base_segmentor.py

from abc import ABCMeta, abstractmethod

from src.base.base_module import BaseModule


class BaseSegmentor(BaseModule, metaclass=ABCMeta):
    """Base class for seg."""

    def __init__(self, init_cfg=None):
        super(BaseSegmentor, self).__init__(init_cfg)


    @property
    def with_backbone(self):
        """bool: whether the segmentor has neck"""
        return hasattr(self, 'backbone') and self.backbone is not None

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
        """bool: whether the segmentor has loss"""
        return hasattr(self, 'loss') and self.loss is not None

    @property
    def with_auxiliary_loss(self):
        """bool: whether the segmentor has auxiliary loss"""
        return hasattr(self, 'auxiliary_loss') and self.auxiliary_loss is not None

    @abstractmethod
    def forward(self, imgs, targets=None, mode='infer',  epoch_num=0, step_num=0, **kwargs):
        pass