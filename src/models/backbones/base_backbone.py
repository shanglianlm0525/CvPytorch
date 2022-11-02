# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/26 17:05
# @Author : liumin
# @File : base_backbone.py

from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from src.base.base_module import BaseModule


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    def __init__(self, subtype=None, in_channels=3, out_channels=[64, 128, 256, 512, 1024],
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'),
                 out_stages=[2, 3, 4], output_stride=32, backbone_path=None, pretrained=False,
                 classifier=False, num_classes=1000, frozen_stages=-1, norm_eval=False):
        super(BaseBackbone, self).__init__()
        self.subtype = subtype
        self.in_channels = in_channels
        self.out_channels = out_channels
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

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        self.build_stage_layer()

        if self.classifier:
            self.classify = self.build_classify_layer()
            self.layers.append(f'classify')

    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        pass

    '''
    stage = []
    stage += self.build_stage_layer(idx, setting)
    self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
    self.layers.append(f'stage{idx + 1}')
    '''


    @abstractmethod
    def build_classify_layer(self):
        """Build a stem layer."""
        pass


    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
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
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)