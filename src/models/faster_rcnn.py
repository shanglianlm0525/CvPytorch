# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/18 9:43
# @Author : liumin
# @File : faster_rcnn.py

import torch
import torch.nn as nn
import torchvision
from torchvision import ops
from torchvision.models import detection
from torchvision.models.detection.rpn import AnchorGenerator

"""
    Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
    https://arxiv.org/pdf/1506.01497.pdf
"""

class FasterRCNN(nn.Module):
    def __init__(self, dictionary=None):
        super(FasterRCNN, self).__init__()

        self.dictionary = dictionary
        self.input_size = [512, 512]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        # load a pre-trained model for topformer and return
        # only the features
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        # FasterRCNN needs to know the number of
        # output channels in a backbone. For mobilenet_v2, it's 1280
        # so we need to add it here
        backbone.out_channels = 1280

        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))

        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be [0]. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)

        # put the pieces together inside a FasterRCNN model
        self.model = detection.FasterRCNN(backbone, num_classes=self.num_classes,
                           rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.5
        if mode == 'infer':
            predicts = self.model(imgs)
            return predicts
        else:
            if mode == 'val':
                losses = {}
                losses['loss'] = 0
                outputs = self.model(imgs)
                return losses, outputs
            else:
                losses = self.model(imgs, targets)
                losses['loss'] = sum(loss for loss in losses.values())
                return losses