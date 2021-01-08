# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/31 11:04
# @Author : liumin
# @File : nanodet_head.py
from functools import partial

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from ..layers.integral import Integral
from ..modules.convs import DepthwiseConvModule, PointwiseConvModule
from ..modules.init_weights import normal_init
from ...losses.det.det_utils import distance2bbox, bbox2distance
from ...losses.det.focal_loss import QualityFocalLoss, DistributionFocalLoss
from ...losses.det.iou_loss import GIoULoss, bbox_overlaps


def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.true_divide(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class NanodetLoss(nn.Module):
    def __init__(self, num_classes, strides=[8, 16, 32], anchor_scales=[8], anchor_ratios=[1.0], reg_max=7, use_sigmoid=True):
        super(NanodetLoss, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.reg_max = reg_max
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        # self.target_layer = GenTargets2()

        self.distribution_project = Integral(self.reg_max)
        self.loss_qfl = QualityFocalLoss(use_sigmoid=use_sigmoid, beta=2.0, loss_weight=1.0)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_bbox = GIoULoss(loss_weight=2.0)


    def get_anchors(self, featmap_sizes, img_shapes, device='cuda'):  # checked!
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_shapes (h,w): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_shapes)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_shape in enumerate(img_shapes):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_shape
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w),device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list


    def forward_single(self, anchors, cls_score, bbox_pred, labels,
                    label_weights, bbox_targets, stride, num_total_samples):

        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4 * (self.reg_max + 1))
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = torch.nonzero((labels >= 0)
                                 & (labels < bg_class_ind), as_tuple=False).squeeze(1)

        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]  # (n, 4 * (reg_max + 1))  ！！！！！NAN？？？？？？
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride

            weight_targets = cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_corners = self.distribution_project(pos_bbox_pred)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,pos_bbox_pred_corners)
            pos_decode_bbox_targets = pos_bbox_targets / stride
            score[pos_inds] = bbox_overlaps(pos_decode_bbox_pred.detach(),pos_decode_bbox_targets,is_aligned=True)
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)
            target_corners = bbox2distance(pos_anchor_centers,
                                           pos_decode_bbox_targets,
                                           self.reg_max).reshape(-1)

            # regression loss
            loss_bbox = self.loss_bbox(pos_decode_bbox_pred,pos_decode_bbox_targets,weight=weight_targets,avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(pred_corners,target_corners,weight=weight_targets[:, None].expand(-1, 4).reshape(-1),avg_factor=4.0)
        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).cuda()

        # qfl loss
        loss_qfl = self.loss_qfl(cls_score, (labels, score), weight=label_weights, avg_factor=num_total_samples)

        return loss_qfl, loss_bbox, loss_dfl, weight_targets.sum()


    def forward(self, preds, gt_meta):
        cls_scores, bbox_preds = preds

        gt_bboxes = gt_meta['gt_bboxes']
        gt_labels = gt_meta['gt_labels']

        # predicts = self.target_layer([preds, gt_bboxes, gt_labels])
        # print('predicts', predicts)
        '''
        losses_qfl, losses_bbox, losses_dfl, \
        avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.strides,
            num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        if avg_factor <= 0:
            loss_qfl = torch.tensor(0, dtype=torch.float32).cuda()
            loss_bbox = torch.tensor(0, dtype=torch.float32).cuda()
            loss_dfl = torch.tensor(0, dtype=torch.float32).cuda()
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

            loss_qfl = sum(losses_qfl)
            loss_bbox = sum(losses_bbox)
            loss_dfl = sum(losses_dfl)

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(
            loss_qfl=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl)

        return loss, loss_states
    '''




class NanodetHead(nn.Module):
    def __init__(self, num_classes=80, input_channels=[96, 96, 96], feat_channels=96, stacked_convs=2, reg_max=7, share_cls_reg=True, use_sigmoid=True):
        super(NanodetHead, self).__init__()
        self.num_classes = num_classes
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.input_channels = input_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.reg_max = reg_max
        self.share_cls_reg = share_cls_reg

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for input_channel in self.input_channels:
            cls_convs = nn.ModuleList()
            reg_convs = nn.ModuleList()

            for i in range(self.stacked_convs):
                chn = input_channel if i == 0 else self.feat_channels
                cls_convs.append(DepthwiseConvModule(chn,chn,3,norm='BatchNorm2d',activation='ReLU'))
                cls_convs.append(PointwiseConvModule(chn,self.feat_channels,3,norm='BatchNorm2d',activation='ReLU'))
                if not self.share_cls_reg:
                    reg_convs.append(DepthwiseConvModule(chn, chn, 3, norm='BatchNorm2d', activation='ReLU'))
                    reg_convs.append(PointwiseConvModule(chn, self.feat_channels, 3, norm='BatchNorm2d', activation='ReLU'))

            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([nn.Conv2d(self.feat_channels,
                                                self.cls_out_channels + 4 * (self.reg_max + 1) if self.share_cls_reg else self.cls_out_channels,
                                                1,padding=0) for _ in self.input_channels])
        self.gfl_reg = nn.ModuleList([nn.Conv2d(self.feat_channels,4 * (self.reg_max + 1),1,padding=0) for _ in self.input_channels])

    def init_weights(self):
        for seq in self.cls_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        for seq in self.reg_convs:
            for m in seq:
                normal_init(m.depthwise, std=0.01)
                normal_init(m.pointwise, std=0.01)
        bias_cls = -4.595  # 用0.01的置信度初始化
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single,
                           feats,
                           self.cls_convs,
                           self.reg_convs,
                           self.gfl_cls,
                           self.gfl_reg,)

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg):
        cls_feat = x
        reg_feat = x
        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)
        if self.share_cls_reg:
            feat = gfl_cls(cls_feat)
            cls_score, bbox_pred = torch.split(feat, [self.cls_out_channels, 4 * (self.reg_max + 1)], dim=1)
        else:
            cls_score = gfl_cls(cls_feat)
            bbox_pred = gfl_reg(reg_feat)
        return cls_score, bbox_pred

