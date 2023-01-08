# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:24
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .convnext import ConvNeXt
from .custom_cspnet import CustomCspNet
from .efficientnet import EfficientNet
from .efficientnet_lite import EfficientNetLite
from .lfd_resnet import LFDResNet
from .lspnet_backbone import LSPNetBackbone
from .mobilenet_v3 import MobileNetV3
from .regnet import RegNet
from .repvgg import RepVGG
from .sgcpnet_backbone import SGCPNetBackbone
from .vgg import VGG
from .resnext import ResNeXt
from .vision_transformer import VisionTransformer
from .wide_resnet import WideResNet
from .squeezenet import SqueezeNet
from .mobilenet_v2 import MobileNetV2
from .shufflenet_v2 import ShuffleNetV2
from .densenet import Densenet

# from .ghostnet import GhostNet



# Image Classification


# Semantic Segmentation
from .seg import MSCAN
from .seg import STDCNet
from .seg import TopFormerBackbone
from .seg import RegSegBackbone
from .seg import ResNet
from .seg import MixVisionTransformer
from .seg import IncepTransformer

# Object Detectiton
from .det import CSPDarknet
from .det import YOLOv5CSPDarknet
from .det import YOLOXCSPDarknet
from .det import YOLOv6EfficientRep
from .det import YOLOv7CSPVoVNet
from .det import YOLOXPAIEfficientRep


__all__ = ['VGG', 'ResNet', 'ResNeXt', 'WideResNet', 'SqueezeNet', 'MobileNetV2', 'MobileNetV3', 'ShuffleNetV2',
            'VisionTransformer', 'ConvNeXt', 'EfficientNet', 'RegNet', 'MSCAN', 'IncepTransformer',

           'YOLOv5CSPDarknet', 'YOLOXCSPDarknet', 'YOLOv6EfficientRep', 'YOLOv7CSPVoVNet', 'YOLOXPAIEfficientRep',
           'STDCNet', 'RepVGG', 'EfficientNetLite', 'CustomCspNet',
           'CSPDarknet', 'RegNet', 'LFDResNet',  'LSPNetBackbone']


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
    elif name == 'MSCAN':
        return MSCAN(**backbone_cfg)
    elif name == 'LSPNetBackbone':
        return LSPNetBackbone(**backbone_cfg)
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'TopFormerBackbone':
        return TopFormerBackbone(**backbone_cfg)
    elif name == 'MixVisionTransformer':
        return MixVisionTransformer(**backbone_cfg)
    elif name == 'IncepTransformer':
        return IncepTransformer(**backbone_cfg)

    # no pretrained weight
    elif name == 'EfficientNetLite':
        return EfficientNetLite(**backbone_cfg)
    elif name == 'CustomCspNet':
        return CustomCspNet(**backbone_cfg)
    elif name == 'CSPDarknet':
        return CSPDarknet(**backbone_cfg)
    elif name == 'SGCPNetBackbone':
        return SGCPNetBackbone(**backbone_cfg)
    elif name == 'LFDResNet':
        return LFDResNet(**backbone_cfg)
    elif name == 'YOLOv5CSPDarknet':
        return YOLOv5CSPDarknet(**backbone_cfg)
    elif name == 'YOLOXCSPDarknet':
        return YOLOXCSPDarknet(**backbone_cfg)
    elif name == 'YOLOv6EfficientRep':
        return YOLOv6EfficientRep(**backbone_cfg)
    elif name == 'YOLOv7CSPVoVNet':
        return YOLOv7CSPVoVNet(**backbone_cfg)
    elif name == 'YOLOXPAIEfficientRep':
        return YOLOXPAIEfficientRep(**backbone_cfg)
    elif name == 'RepVGG':
        return RepVGG(**backbone_cfg)
    elif name == 'RegSegBackbone':
        return RegSegBackbone(**backbone_cfg)

    else:
        raise NotImplementedError(name)
