# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:24
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .convnext import ConvNeXt
from .csp_darknet import CspDarkNet
from .custom_cspnet import CustomCspNet
from .efficientnet import EfficientNet
from .efficientnet_lite import EfficientNetLite
from .efficientrep import EfficientRep
from .lfd_resnet import LFDResNet
from .lspnet_backbone import LSPNetBackbone
from .mobilenet_v3 import MobileNetV3
from .pai_yolox_backbone import PAI_YOLOXBackbone
from .regnet import RegNet
from .regseg_backbone import RegSegBackbone
from .repvgg import RepVGG
from .sgcpnet_backbone import SGCPNetBackbone
from .topformer_backbone import TopFormerBackbone
from .vgg import VGG
from .resnet import ResNet
from .resnext import ResNeXt
from .vision_transformer import VisionTransformer
from .wide_resnet import WideResNet
from .squeezenet import SqueezeNet
from .mobilenet_v2 import MobileNetV2
from .shufflenet_v2 import ShuffleNetV2
from .densenet import Densenet
from .stdcnet import STDCNet
# from .ghostnet import GhostNet
from .yolov5_backbone import YOLOv5Backbone
from .yolov7_backbone import YOLOv7Backbone

__all__ = ['VGG', 'ResNet', 'ResNeXt', 'WideResNet', 'SqueezeNet', 'MobileNetV2', 'MobileNetV3', 'ShuffleNetV2',
            'VisionTransformer', 'ConvNeXt', 'EfficientNet', 'RegNet',

           'YOLOv5Backbone', 'YOLOv7Backbone', 'PAI_YOLOXBackbone', 'STDCNet', 'RepVGG', 'EfficientNetLite', 'CustomCspNet',
           'CspDarkNet', 'RegNet', 'LFDResNet', 'EfficientRep', 'LSPNetBackbone']


def build_backbone(cfg):
    backbone_cfg = deepcopy(cfg)
    name = backbone_cfg.pop('name')

    # default torch pretrained
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
    elif name == 'MobileNetV3':
        return MobileNetV3(**backbone_cfg)
    elif name == 'ShuffleNetV2':
        return ShuffleNetV2(**backbone_cfg)
    elif name == 'EfficientNet':
        return EfficientNet(**backbone_cfg)
    elif name == 'RegNet':
        return RegNet(**backbone_cfg)
    elif name == 'ConvNeXt':
        return ConvNeXt(**backbone_cfg)
    elif name == 'VisionTransformer':
        return VisionTransformer(**backbone_cfg)

    # extra pretrained weight
    elif name == 'STDCNet':
        return STDCNet(**backbone_cfg)
    elif name == 'LSPNetBackbone':
        return LSPNetBackbone(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'TopFormerBackbone':
        return TopFormerBackbone(**backbone_cfg)

    # no pretrained weight
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'CustomCspNet':
        return CustomCspNet(**backbone_cfg)
    elif name == 'CspDarkNet':
        return CspDarkNet(**backbone_cfg)
    elif name == 'EfficientRep':
        return EfficientRep(**backbone_cfg)
    elif name == 'SGCPNetBackbone':
        return SGCPNetBackbone(**backbone_cfg)
    elif name == 'LFDResNet':
        return LFDResNet(**backbone_cfg)
    elif name == 'YOLOv5Backbone':
        return YOLOv5Backbone(**backbone_cfg)
    elif name == 'YOLOv7Backbone':
        return YOLOv7Backbone(**backbone_cfg)
    elif name == 'PAI_YOLOXBackbone':
        return PAI_YOLOXBackbone(**backbone_cfg)
    elif name == 'RepVGG':
        return RepVGG(**backbone_cfg)
    elif name == 'RegSegBackbone':
        return RegSegBackbone(**backbone_cfg)

    else:
        raise NotImplementedError(name)
