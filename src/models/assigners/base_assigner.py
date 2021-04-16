# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/29 10:08
# @Author : liumin
# @File : base_assigner.py

from abc import ABCMeta, abstractmethod


class BaseAssigner(metaclass=ABCMeta):

    @abstractmethod
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        pass