# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/25 18:27
# @Author : liumin
# @File : ssd.py
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import build_backbone

from itertools import product
import numpy as np
import torch
from math import sqrt

from .detects.ssd_detect import PostProcessor


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio


    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class SSDLoss(nn.Module):
    def __init__(self, neg_pos_ratio):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(SSDLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class SSD(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super().__init__()
        self.dictionary = dictionary
        self.input_size = [300, 300]

        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        backbone_cfg = {'name': 'VGG', 'subtype': 'vgg16', 'out_stages': [3, 4],
                        'output_stride': 32, 'pretrained': True }
        self.backbone = build_backbone(backbone_cfg)
        self.backbone.layer2_pool[0].ceil_mode = True
        self.backbone_plus = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.extras_layers  = self.add_extras()

        self.loc, self.conf = self.build_box_head()

        self.post_processor = PostProcessor(top_k=100, nms_thresh=0.01, conf_thresh=0.45)

        self.l2_norm = L2Norm(512, scale=20)
        self.criterion = SSDLoss(neg_pos_ratio=3)

        self.init_params()

    def build_box_head(self):
        n_default_anchors = [4, 6, 6, 6, 4, 4]
        out_channels = [512, 1024, 512, 256, 256, 256]
        loc = []
        conf = []
        for nd, oc in zip(n_default_anchors, out_channels):
            loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, stride=1, padding=1))
            conf.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, stride=1, padding=1))
        return nn.ModuleList(loc), nn.ModuleList(conf)


    def add_extras(self):
        in_channels = [1024, 512, 256, 256]
        out_channels = [512, 256, 256, 256]
        extras_layers = []
        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            mid_channels = out_channel // 2
            if i < 2:
                layer = nn.Sequential(
                    nn.Conv2d(in_channel, mid_channels, kernel_size=1),
                    nn.Conv2d(mid_channels, out_channel, kernel_size=3, stride=2, padding=1)
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(in_channel, mid_channels, kernel_size=1),
                    nn.Conv2d(mid_channels, out_channel, kernel_size=3, stride=1, padding=0)
                )
            extras_layers.append(layer)
        return nn.ModuleList(extras_layers)


    def init_params(self):
        for m in self.backbone_plus.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.extras_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.loc.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.conf.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def trans_specific_format(self, imgs, targets):
        new_boxes = []
        new_labels = []
        new_scales = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_boxes.append(target['boxes'])
            new_labels.append(target['labels'])
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["boxes"] = torch.stack(new_boxes)
        t_targets["labels"] = torch.stack(new_labels)
        t_targets["scales"] = torch.stack(new_scales) if len(new_scales) > 0 else []
        t_targets["height"] = torch.stack(new_heights)
        t_targets["width"] = torch.stack(new_widths)
        t_targets["anchors"] = targets[0]["anchors"]
        return imgs, t_targets

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        '''
        imgs 32 * 3* 300 * 300
        labels
        boxes 32 * 8732 *4
        labels 32*8732
        '''
        imgs, targets = self.trans_specific_format(imgs, targets)

        features = []
        # imgs = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/images.pth')
        # targets1 = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/targets.pth')
        feat1, feat2 = self.backbone(imgs)
        feat1 = self.l2_norm(feat1)
        feat_tmp = feat2 = self.backbone_plus(feat2)
        features.append(feat1)
        features.append(feat2)
        for extras_layer in self.extras_layers:
            feat_tmp = extras_layer(feat_tmp)
            features.append(feat_tmp)


        # features1 = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/features.pth')
        cls_logits = list()
        box_logits = list()
        # apply multibox head to source layers
        for (x, c, l) in zip(features, self.conf, self.loc):
            cls_logits.append(c(x).permute(0, 2, 3, 1).contiguous())
            box_logits.append(l(x).permute(0, 2, 3, 1).contiguous())

        batch_size = imgs.shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.num_classes)
        box_logits = torch.cat([l.view(l.shape[0], -1) for l in box_logits], dim=1).view(batch_size, -1, 4)

        # detections = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/detections.pth')
        # detector_losses = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/detector_losses.pth')

        if mode == 'infer':
            out = []
            # locs: nbatch x 8732 x nlocs, confs: nbatch x 8732 x nlabels, self._priors: 8732 x nlocs
            detections = self.post_processor(cls_logits, box_logits, targets['anchors'])

            return out
        else:
            losses = {}
            '''
            cls_logits = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/cls_logits.pth')
            box_logits = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/bbox_pred.pth')
            targets_boxes = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/gt_boxes.pth')
            targets_labels = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/gt_labels.pth')

            losses['reg_loss1'], losses['cls_loss1'] = self.criterion(cls_logits, box_logits, targets_labels,
                                                                    targets_boxes)
            print(losses)
            '''
            losses['reg_loss'], losses['cls_loss'] = self.criterion(cls_logits, box_logits, targets['labels'], targets['boxes'])
            losses['loss'] = losses['reg_loss'] + losses['cls_loss']

            if mode == 'val':
                outputs = []
                detections = self.post_processor(cls_logits, box_logits, targets['anchors'])
                # print('detections', detections)
                for detection, width, height, scale in zip(detections, targets['width'], targets['height'], targets['scales']):
                    boxes_np = detection['boxes'].cpu().numpy()

                    width = width.cpu().numpy()
                    height = height.cpu().numpy()
                    scale = scale.cpu().numpy()
                    boxes_np[:, 0::2] /= scale[1]
                    boxes_np[:, 1::2] /= scale[0]
                    # clip boxes
                    boxes_np[:, 0::2] = boxes_np[:, 0::2].clip(0, width)
                    boxes_np[:, 1::2] = boxes_np[:, 1::2].clip(0, height)
                    outputs.append({"boxes": torch.tensor(boxes_np),
                                    "labels": detection['labels'], "scores": detection['scores']})
                return losses, outputs
            else:
                return losses