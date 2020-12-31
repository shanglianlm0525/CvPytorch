# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:24
# @Author : liumin
# @File : __init__.py

from copy import deepcopy
from .vgg import VGG
from .resnet import ResNet
from .resnext import ResNeXt
from .wide_resnet import WideResNet
from .squeezenet import SqueezeNet
from .mobilenet_v2 import MobileNetV2
from .shufflenet_v2 import ShuffleNetV2
from .densenet import Densenet
# from .ghostnet import GhostNet

__all__ = ['VGG','ResNet','ResNeXt','WideResNet','SqueezeNet','MobileNetV2','ShuffleNetV2']


def build_backbone(cfg):
    backbone_cfg = deepcopy(cfg)
    name = backbone_cfg.pop('name')
    if name == 'VGG':
        return VGG(**backbone_cfg)
    elif name == 'ResNet':
        return ResNet(**backbone_cfg)
    elif name == 'ResNeXt':
        return ResNeXt(**backbone_cfg)
    elif name == 'WideResNet':
        return WideResNet(**backbone_cfg)
    elif name == 'SqueezeNet':
        return SqueezeNet(**backbone_cfg)
    elif name == 'MobileNetV2':
        return MobileNetV2(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    else:
        raise NotImplementedError
