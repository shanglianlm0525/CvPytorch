# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/10/21 9:13
# @Author : liumin
# @File : efficientdet.py

import torch
import torch.nn as nn
import torchvision

from src.losses.efficientdet_loss import EfficientDetLoss
from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.models.necks import build_neck

"""
    EfficientDet: Scalable and Efficient Object Detection
    https://arxiv.org/pdf/1911.09070.pdf
"""


def getRegressBoxes(anchors, regression):
    """
    decode_box_outputs adapted from https://github.com/google/automl/blob/master/efficientdet/anchors.py

    Args:
        anchors: [batchsize, boxes, (y1, x1, y2, x2)]
        regression: [batchsize, boxes, (dy, dx, dh, dw)]

    Returns:

    """
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]

    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha

    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a

    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.

    return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class EfficientDet(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(EfficientDet, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 512, 512)

        self.num_classes = 80 # len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        # 'efficient0'
        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = self.build_necks()
        self.head = build_head(self.model_cfg.HEAD)

        self.loss = EfficientDetLoss()

        self.conf_thres = 0.05  # confidence threshold
        self.iou_thres = 0.5  # NMS IoU threshold

    def build_necks(self):
        neck_list= []
        for _ in range(self.fpn_cell_repeat):
            self.model_cfg.NECK.__setitem__('in_channels', self.conv_channel_coef)
            self.model_cfg.NECK.__setitem__('out_channels', self.fpn_num_filter)
            self.model_cfg.NECK.__setitem__('first_time', True if _ == 0 else False)
            self.model_cfg.NECK.__setitem__('attention', True if self.compound_coef < 6 else False)
            neck_list.append(build_neck(self.model_cfg.NECK))
        return nn.Sequential(*neck_list)

    def setup_extra_params(self):
        self.compound_coef = int(self.model_cfg.TYPE[-1])

        self.conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576]
        }[self.compound_coef]

        self.fpn_num_filter = [64, 88, 112, 160, 224, 288, 384, 384][self.compound_coef]
        self.fpn_cell_repeat = [3, 4, 5, 6, 7, 7, 8, 8][self.compound_coef]

        box_class_repeat = [3, 3, 3, 4, 4, 4, 5, 5][self.compound_coef]
        pyramid_level =[5, 5, 5, 5, 5, 5, 5, 5][self.compound_coef]
        anchor_scale = [4., 4., 4., 4., 4., 4., 4., 5.][self.compound_coef]

        ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
        scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
        num_anchors = len(ratios) * len(scales)

        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)
        self.model_cfg.HEAD.__setitem__('fpn_num_filter', self.fpn_num_filter)
        self.model_cfg.HEAD.__setitem__('box_class_repeat', box_class_repeat)
        self.model_cfg.HEAD.__setitem__('pyramid_level', pyramid_level)
        self.model_cfg.HEAD.__setitem__('num_anchors', num_anchors)
        self.model_cfg.HEAD.__setitem__('anchor_scale', anchor_scale)


    def trans_specific_format(self, imgs, targets):
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []

        max_num_annos = max(target['labels'].shape[0] for target in targets)
        new_gts = torch.ones((len(targets), max_num_annos, 5), device=targets[0]['labels'].device) * -1
        for idx, target in enumerate(targets):
            new_gts[idx, :target['labels'].shape[0], :] = torch.cat([target['boxes'], target['labels'].unsqueeze(1)],1).unsqueeze(0)
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["gts"] = new_gts
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets


    def forward(self, imgs, targets=None, mode='infer', **kwargs):

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)
            b, _, height, width = imgs.shape
            """
                torch.Size([12, 64, 64, 64]) torch.Size([12, 64, 32, 32]) torch.Size([12, 64, 16, 16])
                torch.Size([12, 49104, 4]) torch.Size([12, 49104, 80]) torch.Size([1, 49104, 4])
            """
            losses = {}
            features = self.backbone(imgs)
            features = self.neck(features)
            train_out = self.head(imgs, features)

            losses['cls_loss'], losses['box_loss'] = self.loss(train_out, targets["gts"])
            losses['loss'] = losses['cls_loss'] + losses['box_loss']

            if mode == 'val':
                outputs = []
                t_classifications, t_regressions, t_anchors = train_out
                transformed_anchors = getRegressBoxes(t_anchors, t_regressions)

                t_scores = torch.max(t_classifications, dim=2, keepdim=True)[0]
                scores_over_thresh = (t_scores > self.conf_thres)[:, :, 0]
                for i, (width, height, scale, pad, classification, t_anchor, score) in enumerate(zip(targets['width'], targets['height'], targets['scales'], targets['pads'],
                                                                        t_classifications, transformed_anchors, t_scores)):

                    if scores_over_thresh[i].sum() == 0:
                        outputs.append({"boxes": torch.empty([0, 4]), "labels": torch.empty([0]), "scores": torch.empty([0])})
                        continue

                    classification_per = classification[scores_over_thresh[i]].permute(1, 0)
                    transformed_anchors_per = t_anchor[scores_over_thresh[i]]
                    score_per = score[scores_over_thresh[i]]
                    scores_, classes_ = classification_per.max(dim=0)
                    anchors_nms_idx = torchvision.ops.boxes.batched_nms(transformed_anchors_per, score_per[:, 0], classes_, iou_threshold=self.iou_thres)
                    # print(anchors_nms_idx)

                    if anchors_nms_idx.shape[0] == 0:
                        outputs.append({"boxes": torch.empty([0, 4]), "labels": torch.empty([0]), "scores": torch.empty([0])})
                        continue

                    labels = classes_[anchors_nms_idx]
                    scores = scores_[anchors_nms_idx]
                    bboxes = transformed_anchors_per[anchors_nms_idx, :]

                    scale = scale.cpu().numpy()
                    pad = pad.cpu().numpy()
                    width = width.cpu().numpy()
                    height = height.cpu().numpy()

                    bboxes_np = bboxes.cpu().detach().numpy()
                    bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                    bboxes_np[:, [1, 3]] -= pad[0]
                    bboxes_np[:, [0, 2]] /= scale[1]
                    bboxes_np[:, [1, 3]] /= scale[0]

                    # clip boxes
                    bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                    bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)
                    outputs.append({"boxes": torch.tensor(bboxes_np), "labels": labels, "scores": scores})

                return losses, outputs
            else:
                return losses