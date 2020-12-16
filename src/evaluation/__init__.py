# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/16 9:35
# @Author : liumin
# @File : __init__.py

from .eval_classification import ClassificationEvaluator
from .eval_detection import VOCEvaluator, COCOEvaluator
from .eval_segmentation import SegmentationEvaluator

__all__ = ['ClassificationEvaluator', 'VOCEvaluator', 'COCOEvaluator','SegmentationEvaluator']

def build_evaluator(cfg, dataset):
    'Using adapter design patterns'
    if cfg.evaluator.name == 'classification':
        return ClassificationEvaluator(dataset)
    elif cfg.evaluator.name == 'voc_detection':
        return VOCEvaluator(dataset)
    elif cfg.evaluator.name == 'coco_detection':
        return COCOEvaluator(dataset)
    elif cfg.evaluator.name == 'segmentation':
        return SegmentationEvaluator(dataset)
    else:
        raise NotImplementedError
