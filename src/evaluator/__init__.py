# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/16 9:35
# @Author : liumin
# @File : __init__.py

from .eval_classification import ClassificationEvaluator
from .eval_detection import VOCEvaluator
from .eval_segmentation import SegmentationEvaluator
from .eval_coco import CocoEvaluator

__all__ = [
    'ClassificationEvaluator',
    'VOCEvaluator',
    'SegmentationEvaluator',
    'CocoEvaluator']

def build_evaluator(cfg, dataset):
    'Using adapter design patterns'
    if cfg.EVALUATOR.NAME == 'classification':
        return ClassificationEvaluator(dataset)
    elif cfg.EVALUATOR.NAME == 'voc_detection':
        return VOCEvaluator(dataset)
    elif cfg.EVALUATOR.NAME.startswith('coco'):
        if cfg.EVALUATOR.NAME == 'coco_segmentation':
            iou_types = ["segm"]
        elif cfg.EVALUATOR.NAME == 'coco_detection':
            iou_types = ["bbox"]
        elif cfg.EVALUATOR.NAME == 'coco_keypoints':
            iou_types = ["bbox", "keypoints"]
        else: # 'coco_instance'
            iou_types = ["bbox", "segm"]
        return CocoEvaluator(dataset, iou_types)
    elif cfg.EVALUATOR.NAME == 'segmentation':
        return SegmentationEvaluator(dataset)
    else:
        raise NotImplementedError
