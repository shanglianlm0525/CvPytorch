# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/27 15:23
# @Author : liumin
# @File : yolov5_csp_darknet.py

import torch
import torch.nn as nn
import math
from torch.nn.modules.batchnorm import _BatchNorm

from src.models.backbones.det.base_yolo_backbone import BaseYOLOBackbone
from src.models.bricks import ConvModule
from src.models.modules.yolo_modules import CSPLayer, SPPF


class YOLOv5CSPDarknet(BaseYOLOBackbone):
    cfg = {"n": [0.33, 0.25],
            "t": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='cspdark_s', in_channels=3, out_channels=[64, 128, 256, 512, 1024], num_blocks=[3, 6, 9, 3], spp_ksizes=5, depthwise=False,
                 conv_cfg=None, norm_cfg=dict(type='BN', momentum=0.03, eps=0.001), act_cfg=dict(type='SiLU', inplace=True),
                 out_stages=[2, 3, 4], output_stride=32, backbone_path=None, pretrained=False,
                 frozen_stages=-1, norm_eval=False):
        self.spp_ksizes = spp_ksizes

        super(YOLOv5CSPDarknet, self).__init__(subtype=subtype, cfg=self.cfg, in_channels=in_channels, out_channels=out_channels, num_blocks=num_blocks, depthwise=depthwise,
                                               conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, output_stride=output_stride,
                                               out_stages = out_stages, pretrained = pretrained, backbone_path = backbone_path,
                                               frozen_stages = frozen_stages,norm_eval = norm_eval)


    def build_stem_layer(self):
        """Build a stem layer."""
        return ConvModule(
            self.in_channels,
            self.out_channels[0],
            kernel_size=6,
            stride=2,
            padding=2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)


    def build_stage_layer(self, idx, in_channels, out_channels, num_blocks):
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            n=num_blocks,
            shortcut=False if idx == 3 else True,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        stage.append(csp_layer)
        if idx == 3:
            spp = SPPF(
                out_channels,
                out_channels,
                kernel_sizes=self.spp_ksizes,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03


    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


    def train(self, mode=True):
        super(YOLOv5CSPDarknet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


if __name__ == "__main__":
    model = YOLOv5CSPDarknet('cspdark_s')
    print(model)

    input = torch.randn(1, 3, 640, 640)
    out = model(input)
    for o in out:
        print(o.shape)