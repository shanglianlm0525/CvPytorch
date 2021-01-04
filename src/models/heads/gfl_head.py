# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 18:04
# @Author : liumin
# @File : gfl_head.py

import torch
import torch.nn as nn
import numpy as np

from ..layers.integral import Integral

"""
    Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection.
    https://arxiv.org/abs/2006.04388
"""

class GFLHead(nn.Module):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
        Boxes for Dense Object Detection.

        GFL head structure is similar with ATSS, however GFL uses
        1) joint representation for classification and localization quality, and
        2) flexible General distribution for bounding box locations,
        which are supervised by
        Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

        https://arxiv.org/abs/2006.04388

        Args:
            num_classes (int): Number of categories excluding the background
                category.
            in_channels (int): Number of channels in the input feature map.
            stacked_convs (int): Number of conv layers in cls and reg tower.
                Default: 4.
            conv_cfg (dict): dictionary to construct and config conv layer.
                Default: None.
            norm_cfg (dict): dictionary to construct and config norm layer.
                Default: dict(type='GN', num_groups=32, requires_grad=True).
            loss_qfl (dict): Config of Quality Focal Loss (QFL).
            reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
                in QFL setting. Default: 16.
        Example:
            >>> self = GFLHead(11, 7)
            >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
            >>> cls_quality_score, bbox_pred = self.forward(feats)
            >>> assert len(cls_quality_score) == len(self.scales)
        """
    def __init__(self,
                 num_classes,
                 loss,
                 input_channel,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 reg_max=16,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.reg_max = reg_max
        use_sigmoid = True
        octave_scales = np.array([2 ** (i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(GFLHead, self).__init__(num_classes, loss, use_sigmoid, input_channel, anchor_scales=anchor_scales, **kwargs)

        self.distribution_project = Integral(self.reg_max)

        # self.loss_qfl = QualityFocalLoss(use_sigmoid=True, beta=2.0, loss_weight=1.0)
        # self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        # self.loss_bbox = GIoULoss(loss_weight=2.0)
        self.init_weights()