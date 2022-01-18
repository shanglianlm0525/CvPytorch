# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/12/16 8:57
# @Author : liumin
# @File : fcos.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.models.necks import build_neck
from src.losses import build_loss
from src.models.detects import build_detect

"""
    FCOS: Fully Convolutional One-Stage Object Detection
    https://arxiv.org/pdf/1904.01355.pdf
"""

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w = batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes


class FCOS(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(FCOS, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 800, 800)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        '''
        backbone_cfg = {'name': 'ResNet', 'subtype': 'resnet50', 'out_stages': [2, 3, 4], 'output_stride': 32,
                        'pretrained': True}
        self.backbone = build_backbone(backbone_cfg)
        self.neck = FPN(256, use_p5=True)
        self.head = ClsCntRegHead(256, class_num=80, GN=True, cnt_on_reg=True, prior=0.01)
        self.detect = Detect(score_threshold=0.05, nms_iou_threshold=0.6, max_detection_boxes_num=1000, strides=[8, 16, 32, 64, 128])
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        self.target_layer = GenTargets(strides=strides, limit_range=limit_range)
        self.loss_layer = LOSS()
        '''
        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)
        self.detect = build_detect(self.model_cfg.DETECT)
        self.loss = build_loss(self.model_cfg.LOSS)
        self.clip_boxes = ClipBoxes()

        self.conf_thres = 0.05  # confidence threshold
        self.iou_thres = 0.6  # NMS IoU threshold

        self.init_weights()

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)


    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = False

    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False

        self.apply(freeze_bn)
        print("INFO===>success frozen BN")
        self.backbone.freeze_stages(1)
        print("INFO===>success frozen backbone stage1")


    def trans_specific_format(self, imgs, targets):
        new_boxes = []
        new_labels = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []

        batch_size=len(imgs)
        h_list = [int(img.shape[1]) for img in imgs]
        w_list = [int(img.shape[2]) for img in imgs]
        num_list = [int(target['labels'].shape[0]) for target in targets]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        max_num = np.array(num_list).max()
        '''
        pad_imgs = []
        for img in imgs:
            pad_imgs.append(
                F.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))
        '''

        for i, target in enumerate(targets):
            new_boxes.append(F.pad(target['boxes'], (0, 0, 0, max_num - target['boxes'].shape[0]), value=-1))
            new_labels.append(F.pad(target['labels'], (0, max_num - target['labels'].shape[0]), value=-1))

            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["boxes"] = torch.stack(new_boxes)
        t_targets["labels"] = torch.stack(new_labels)
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets


    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05
        if mode == 'infer':
            pass
        else:
            losses = {}
            imgs, targets = self.trans_specific_format(imgs, targets)

            C3, C4, C5 = self.backbone(imgs)
            all_P = self.neck([C3, C4, C5])
            cls_logits, cnt_logits, reg_preds = self.head(all_P)
            out = [cls_logits, cnt_logits, reg_preds]
            # out_targets = self.target_layer([out, targets["boxes"], targets["labels"]])
            # loss_tuple = self.loss_layer([out, out_targets])
            loss_tuple = self.loss(out, targets["boxes"], targets["labels"])

            losses['cls_loss'] = loss_tuple[0]
            losses['cnt_loss'] = loss_tuple[1]
            losses['reg_loss'] = loss_tuple[2]
            losses['loss'] = loss_tuple[-1]

            if mode == 'val':
                pred_scores, pred_labels, pred_boxes = self.detect(out)
                pred_boxes = self.clip_boxes(imgs, pred_boxes)

                outputs = []
                for i, (width, height, scale, pad, pred_box, pred_label, pred_score) in enumerate(
                        zip(targets['width'], targets['height'], targets['scales'], targets['pads'],
                            pred_boxes, pred_labels, pred_scores)):

                    scale = scale.cpu().numpy()
                    pad = pad.cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    pred_box = pred_box.clone()

                    bboxes_np = pred_box[:, :4].cpu().numpy()
                    bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                    bboxes_np[:, [1, 3]] -= pad[0]
                    bboxes_np[:, [0, 2]] /= scale[1]
                    bboxes_np[:, [1, 3]] /= scale[0]

                    # clip boxes
                    bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                    bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)

                    keep = pred_score > threshold
                    outputs.append({"boxes": torch.tensor(bboxes_np)[keep], "labels": pred_label[keep], "scores": pred_score[keep]})
                return losses, outputs
            else:
                return losses