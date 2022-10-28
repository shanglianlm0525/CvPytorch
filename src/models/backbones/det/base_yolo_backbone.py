# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/27 15:47
# @Author : liumin
# @File : base_yolo_backbone.py

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from src.base.base_module import BaseModule
from src.models.bricks import DepthwiseSeparableConvModule, ConvModule


class BaseYOLOBackbone(BaseModule, metaclass=ABCMeta):
    def __init__(self, subtype=None, cfg=None, in_channels=3, out_channels=[64, 128, 256, 512, 1024], num_blocks=[3, 6, 9, 3], depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'),
                 out_stages=[2, 3, 4], output_stride=32, backbone_path=None, pretrained=False,
                 classifier=False, num_classes=1000, frozen_stages=-1, norm_eval=False):
        super(BaseYOLOBackbone, self).__init__()
        self.subtype = subtype
        self.cfg = cfg
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.depthwise = depthwise
        self.out_stages = out_stages
        self.output_stride = output_stride
        self.backbone_path = backbone_path
        self.pretrained = pretrained
        self.classifier = classifier
        self.num_classes = num_classes
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        # self.conv = DepthwiseSeparableConvModule if depthwise else ConvModule

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        self.out_channels = list(map(lambda x: int(x * width_mul), self.out_channels))
        self.num_blocks = list(map(lambda x: max(round(x * depth_mul), 1), self.num_blocks))

        self.stem = self.build_stem_layer()
        self.stage_names = ['stem']

        for idx, (in_channels, out_channels, num_blocks) in enumerate(zip(self.out_channels[:-1], self.out_channels[1:], self.num_blocks)):
            self.add_module(f'stage{idx + 1}', nn.Sequential(*self.build_stage_layer(idx, in_channels, out_channels, num_blocks)))
            self.stage_names.append(f'stage{idx + 1}')

        self.init_weights()

    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass


    @abstractmethod
    def build_stage_layer(self, idx, in_channels, out_channels, num_blocks):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass


    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.stages[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def train(self, mode = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, stage_name in enumerate(self.stage_names):
            stage = getattr(self, stage_name)
            x = stage(x)
            if i in self.out_stages:
                outs.append(x)

        return outs