# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/26 11:05
# @Author : liumin
# @File : yolo_series.py

import torch
import torch.nn as nn

from src.models.detectors.single_stage import SingleStageDetector


class YOLODetector(SingleStageDetector):
    def __init__(self, **kwargs):
        super(YOLODetector, self).__init__(**kwargs)


    def setup_extra_params(self):
        self.model_cfg.BACKBONE.__setitem__('subtype', self.model_cfg.TYPE)
        self.model_cfg.NECK.__setitem__('subtype', self.model_cfg.TYPE)
        self.model_cfg.HEAD.__setitem__('subtype', self.model_cfg.TYPE)

    def trans_specific_format(self, imgs, targets):
        max_labels = max([target['labels'].shape[0] for target in targets])
        new_gts = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_gt = torch.zeros((max_labels, 5), device=imgs[0].device)
            gt_tmp = torch.cat([target['labels'].unsqueeze(1), target['boxes']], 1)
            new_gt[:gt_tmp.shape[0], :] = gt_tmp
            new_gts.append(new_gt)
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            if target.__contains__('height'):
                new_heights.append(target['height'])
            if target.__contains__('width'):
                new_widths.append(target['width'])

        t_targets = {}
        t_targets["gt"] = torch.stack(new_gts)
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets
