import cv2
import numpy as np
import importlib
import collections
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from lib.mvutils.utils.config import CommonConfiguration
from .model_config import ModelConfiguration
from .dictionary import ExtensibleDictionary
from ..dataset.transform import ToRGB, GetData
from src.dataset.dataloader import MultiDatasetLoader
import src.dataset.transform as T

def draw_box_with_tag(im, bbox, cls, score, class_list):
    cls_str = class_list[int(cls)]
    tag_height = 12
    tag_width = len(cls_str) * 8 + 30
    _bbox = bbox.astype(np.int32)
    cv2.rectangle(im, (_bbox[0], _bbox[1]), (_bbox[2], _bbox[3]), (0, 0, 255, 255), 2)
    cv2.rectangle(im, (_bbox[0], max(0, _bbox[1] - tag_height)), (_bbox[0] + tag_width, _bbox[1]), (255, 0, 0, 80), -1)
    cv2.putText(im,
        "{} {:.2f}".format(cls_str, score.item()),
        (_bbox[0], max(0, _bbox[1] - 2)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (255, 255, 255, 80),
	 )
    return im

def draw_point_with_tag(im, bbox, center, cls, score, class_list):
    cls_str = class_list[int(cls)]
    tag_height = 12
    tag_width = len(cls_str) * 8 + 30
    _bbox = bbox.astype(np.int32)
    cv2.circle(im, center, 5, (0, 255, 255, 128), -1)
    cv2.rectangle(im, (_bbox[0], _bbox[1] - tag_height), (_bbox[0] + tag_width, _bbox[1]), (255, 0, 0, 80), -1)
    cv2.putText(im,
     "{} {:.2f}".format(cls_str, score.item()),
        (_bbox[0], _bbox[1] - 2),
        cv2.FONT_HERSHEY_DUPLEX,
        0.4,
        (255, 255, 255, 80),
    )
    return im

def draw_result_on_image(im, boxes, scores, classes, class_list):
    for bx, scr, cls in zip(boxes, scores, classes):
        color = (
            min(1 * (1 - scr) / 0.5 + 0 * (min(scr, 0.5) - 0.5) / 0.5, 1),
            max(0 * (scr - 0.5) / 0.5, 0),
            max(1 * (scr - 0.5) / 0.5, 0),
        )
        center = int((bx[2] + bx[0]) / 2), int((bx[3] + bx[1]) / 2)
        cls_str = class_list[int(cls)]
        if cls_str == "carline":
            cv2.circle(im, center, 2, color, -1)
        elif cls_str == "curb":
            cv2.circle(im, center, 2, color)
        elif cls_str == "corners":
            im = draw_point_with_tag(im, bx, center, cls, scr, class_list)
        else:
            im = draw_box_with_tag(im, bx, cls, scr, class_list)

def prepare_model(full_model_class_str, model_cfg, dataset_cfg, model_path="", bitwidth=None):
    *model_mod_str_parts, model_class_str = full_model_class_str.split(".")
    model_class = getattr(importlib.import_module(".".join(model_mod_str_parts)), model_class_str)
    model = model_class(model_cfg, dataset_cfg)

    if model_path:
        chkpnt = torch.load(model_path)
        model.load_state_dict(chkpnt["integrated_model"], strict=False)
    '''
    # set w_bit, b_bit on QConv2d and QLinear
    from ..quantization.modules import QConv2d, QLinear
    if bitwidth is not None:
             for m in model.modules():
            if isinstance(m, QConv2d) or isinstance(m, QLinear):
                m.w_bit = bitwidth.w_bit
                m.b_bit = bitwidth.b_bit
               m.b_ext_bit = bitwidth.b_ext_bit or 0
    '''
    return model

def prepare_model_cfg(model_cfg_base_path, model_cfg_override_dict):
    model_cfg = ModelConfiguration.from_dict({})
    if model_cfg_base_path is not None:
        model_cfg = ModelConfiguration.from_yaml(model_cfg_base_path)
    if model_cfg_override_dict:
        model_cfg_override = ModelConfiguration.from_dict(model_cfg_override_dict)
        model_cfg.update(model_cfg_override)
    return model_cfg

def prepare_dictionary(dictionary_base_path, dictionary_override_dict):
    dictionary = ExtensibleDictionary.from_dict({})
    if dictionary_base_path is not None:
        dictionary = ExtensibleDictionary.from_yaml(dictionary_base_path)
    if dictionary_override_dict:
        dictionary_override = ExtensibleDictionary.from_dict(dictionary_override_dict)
        dictionary.update(dictionary_override)
    return dictionary


def prepare_transforms(transform_cfg):
    train_transforms = []
    val_transforms = []
    eval_transforms = []
    for transform_cls_name, params in deepcopy(transform_cfg).items():
        transform_cls = getattr(T, transform_cls_name)
        assert (
            transform_cls in T.available_transforms
        ), f"{transform_cls.__name__} not in available transforms: {T.available_transforms}"
        try:
            phase = params.pop("phase", "all")
        except AttributeError:  # which means params is None
            params = {}
            phase = "all"
        transform = transform_cls(**params)
        if phase.lower() in ["train", "all", "train_val", "train_val_eval"]:
            train_transforms.append(transform)
        if phase.lower() in ["val", "validation", "validate", "all", "train_val", "train_val_eval", "val_eval"]:
            val_transforms.append(transform)
        if phase.lower() in ["eval", "evaluator", "evaluate", "all", "train_val_eval", "val_eval"]:
            eval_transforms.append(transform)

    train_transforms = [ToRGB()] + train_transforms + [GetData()]
    val_transforms = [ToRGB()] + val_transforms + [GetData()]
    eval_transforms = [ToRGB()] + eval_transforms + [GetData()]
    return {"train": train_transforms, "val": val_transforms, "eval": eval_transforms}


def prepare_infer_only_dataloader(
    full_dataset_class_str,
    dictionary,
    im_dir,
    im_suffix,
    transforms,
    batch_size=1,
    indices=None,
    handle_transforms_within_dataset=False,
):
    *dataset_str_parts, dataset_class_str = full_dataset_class_str.split(".")
    dataset_class = getattr(importlib.import_module(".".join(dataset_str_parts)), dataset_class_str)
    dataset = dataset_class("dataset", dictionary, im_dir=im_dir, im_suffix=im_suffix, transforms=transforms, indices=indices)
    return MultiDatasetLoader(
        [dataset],
        transforms=None  if handle_transforms_within_dataset else [transforms],
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )

'''
def register_input_output_forward_hook_to_model(model):
    for m in model.modules():
        m.register_forward_hook(store_input_output_hook)
'''
