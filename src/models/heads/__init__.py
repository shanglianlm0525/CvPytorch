# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 10:59
# @Author : liumin
# @File : __init__.py

from copy import deepcopy

from .efficientdet_head import EfficientdetHead
from .fastestdet_head import FastestDetHead
from .fcos_head import FCOSHead
from .gflv2_head import GFocalHeadV2
from .lspnet_head import LSPNetHead
from .nanodet_head import NanoDetHead
from .nanodetplus_aux_head import NanoDetPlusAuxHead
from .nanodetplus_head import NanoDetPlusHead
from .openpose_head import OpenPoseHead
from .ppliteseg_head import PPLiteSegHead
from .sgcpnet_head import SGCPNetHead
from .yolop_head import YOLOPHead
from .yolov5_head import YOLOv5Head
from .yolov6_head import YOLOv6Head
from .yolov7_head import YOLOv7Head
# from .tood_head import TOODHead


# Image Classification


# Semantic Segmentation
from .seg import LightHamHead
from .seg import STDCHead
from .seg import FCNHead
from .seg import TopFormerHead
from .seg import RegSegHead
from .seg import Deeplabv3Head
from .seg import Deeplabv3PlusHead
from .seg import PSPHead
from .seg import UPerHead
from .seg import SegFormerHead
from .seg import UperNetAlignHead
from .seg import UpConcatHead

# Object Detectiton
from .det import YOLOXHead
from .det import YOLOv6Effidehead


__all__ = [
    'YOLOv5Head',
    'YOLOXHead',
    'YOLOv7Head',
    # 'TOODHead',
    'NanoDetHead',
    'NanoDetPlusHead',
    'NanoDetPlusAuxHead',
    'YOLOPHead',
    'GFocalHeadV2',
    'FCOSHead',
    'YOLOv6Effidehead',
    'FastestDetHead',
    'Deeplabv3Head',
    'Deeplabv3PlusHead',
    'STDCHead',
    'LightHamHead',
    'LSPNetHead',
    'RegSegHead',
    'OpenPoseHead',
    'SGCPNetHead',
    'PPLiteSegHead',
    'TopFormerHead',
    'FCNHead',
    'PSPHead',
    'SegFormerHead',
    'UpConcatHead'
]


def build_head(cfg):
    head_cfg = deepcopy(cfg)
    name = head_cfg.pop('name')

    if name == 'YOLOv5Head':
        return YOLOv5Head(**head_cfg)
    elif name == 'YOLOXHead':
        return YOLOXHead(**head_cfg)
    elif name == 'YOLOv6Effidehead':
        return YOLOv6Effidehead(**head_cfg)
    elif name == 'YOLOv7Head':
        return YOLOv7Head(**head_cfg)
    # elif name == 'TOODHead':
    #    return TOODHead(**head_cfg)
    elif name == 'FCOSHead':
        return FCOSHead(**head_cfg)
    elif name == 'NanoDetHead':
        return NanoDetHead(**head_cfg)
    elif name == 'NanoDetPlusHead':
        return NanoDetPlusHead(**head_cfg)
    elif name == 'NanoDetPlusAuxHead':
        return NanoDetPlusAuxHead(**head_cfg)
    elif name == 'GFocalHeadV2':
        return GFocalHeadV2(**head_cfg)
    elif name == 'YOLOPHead':
        return YOLOPHead(**head_cfg)
    elif name == 'EfficientdetHead':
        return EfficientdetHead(**head_cfg)
    elif name == 'FastestDetHead':
        return FastestDetHead(**head_cfg)

    elif name == 'Deeplabv3Head':
        return Deeplabv3Head(**head_cfg)
    elif name == 'Deeplabv3PlusHead':
        return Deeplabv3PlusHead(**head_cfg)
    elif name == 'STDCHead':
        return STDCHead(**head_cfg)
    elif name == 'FCNHead':
        return FCNHead(**head_cfg)
    elif name == 'LightHamHead':
        return LightHamHead(**head_cfg)
    elif name == 'LSPNetHead':
        return LSPNetHead(**head_cfg)
    elif name == 'PPLiteSegHead':
        return PPLiteSegHead(**head_cfg)
    elif name == 'RegSegHead':
        return RegSegHead(**head_cfg)
    elif name == 'SGCPNetHead':
        return SGCPNetHead(**head_cfg)
    elif name == 'TopFormerHead':
        return TopFormerHead(**head_cfg)
    elif name == 'PSPHead':
        return PSPHead(**head_cfg)
    elif name == 'UPerHead':
        return UPerHead(**head_cfg)
    elif name == 'SegFormerHead':
        return SegFormerHead(**head_cfg)
    elif name == 'UperNetAlignHead':
        return UperNetAlignHead(**head_cfg)
    elif name == 'UpConcatHead':
        return UpConcatHead(**head_cfg)


    elif name == 'OpenPoseHead':
        return OpenPoseHead(**head_cfg)
    else:
        raise NotImplementedError(name)