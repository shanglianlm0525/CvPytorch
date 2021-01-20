# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/19 12:55
# @Author : liumin
# @File : yolov3_head.py

import torch
import torch.nn as nn
from CvPytorch.src.models.modules.convs import ConvModule

class YOLOV3Head(nn.Module):
    def __init__(self, num_classes=80, in_channels=[128, 256, 512],
                 out_channels=[256, 512, 1024],
                 featmap_strides=[8, 16, 32], num_anchors=3):
        super(YOLOV3Head, self).__init__()
        # Check params
        assert (len(in_channels) == len(out_channels) == len(featmap_strides))
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = num_anchors
        # int: number of attributes in pred_map, bboxes (4) + objectness (1) + num_classes
        self.num_attrib = 5 + self.num_classes

        for i, (in_c, out_c) in enumerate(zip(self.in_channels, self.out_channels)):
            self.add_module(f'conv_bridge{i}',ConvModule(in_c, out_c, kernel_size=3, stride=1,
                       padding=1, dilation=1, groups=1, bias=False, norm='BatchNorm2d', activation='ReLU'))
            self.add_module(f'conv_pred{i}', nn.Conv2d(out_c, self.num_anchors * self.num_attrib, 1))

    def forward(self, feats):
        """Forward features from the upstream network.
            Args:
                feats (tuple[Tensor]): Features from the upstream network, each is
                    a 4D-tensor.
            Returns:
                tuple[Tensor]: A tuple of multi-level predication map, each is a
                    4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == len(self.in_channels)
        pred_maps = []
        for i, x in enumerate(feats):
            conv_bridge = getattr(self, f'conv_bridge{i}')
            conv_pred = getattr(self, f'conv_pred{i}')
            pred_map = conv_pred(conv_bridge(x))
            pred_maps.append(pred_map)

        return tuple(pred_maps)


if __name__ == '__main__':
    in_channels=[128, 256, 512]
    out_channels = [256, 512, 1024]
    scales = [32, 16, 8]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    model = YOLOV3Head(80, in_channels, out_channels)
    print(model)
    outputs = model(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')