# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2023/2/7 14:35
# @Author : liumin
# @File : yolov6_detect.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.losses.det.yolov6_loss import dist2bbox
from src.models.anchors.yolov6_anchor_generator import generate_anchors
from src.models.detects.base_yolo_detect import BaseYOLODetect


class YOLOv6Detect(BaseYOLODetect):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, subtype='yolov6_s', in_channels=[128, 256, 512], num_layers=3, strides=[8, 16, 32],
                 conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='SiLU'), **kwargs):
        super(YOLOv6Detect, self).__init__(subtype=subtype, in_channels=in_channels,
                                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.subtype = subtype
        self.grid = [torch.zeros(1)] * num_layers
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0


    def forward(self, outputs):
        x, cls_score_list, reg_dist_list = outputs
        batch_size = x[0].shape[0]

        anchor_points, stride_tensor = generate_anchors(
            x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

        pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
        pred_bboxes *= stride_tensor
        return torch.cat(
            [
                pred_bboxes,
                torch.ones((batch_size, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                cls_score_list
            ],
            axis=-1)

