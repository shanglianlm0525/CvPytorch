# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/19 13:17
# @Author : liumin
# @File : gflv2_head.py
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import torchvision

from src.losses.det.general_focal_losses import QualityFocalLoss, DistributionFocalLoss
from src.losses.det.iou_losses import GIoULoss
from src.models.assigners.ota_assigner import SimOTAAssigner, bbox_overlaps
from src.models.heads.gflv2_head_bounding_box import BoxList
from src.models.modules.convs import ConvModule
from src.models.modules.init_weights import normal_init


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   iou_thr,
                   max_num=100,
                   score_factors=None):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1)
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)
    scores = multi_scores
    # filter out boxes with low scores
    valid_mask = scores > score_thr   # 1000 * 80 bool

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    # bboxes -> 1000, 4
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)     # mask->  1000*80*4, 80000*4
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero(as_tuple=False)[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        scores = multi_bboxes.new_zeros((0, ))

        return bboxes, scores, labels

    keep = torchvision.ops.batched_nms(bboxes, scores, labels, iou_thr)

    if max_num > 0:
        keep = keep[:max_num]

    return bboxes[keep], scores[keep], labels[keep]



def postprocess_gfocal(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, imgs=None):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        multi_bboxes = image_pred[:, :4]
        multi_scores = image_pred[:, 5:]
        detections, scores, labels = multiclass_nms(multi_bboxes, multi_scores, conf_thre, nms_thre, 500)
        detections = torch.cat((detections, scores[:, None], scores[:, None], labels[:, None]), dim=1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    # transfer to BoxList
    for i in range(len(output)):
        res = output[i]
        if res is None or imgs is None:
            boxlist = BoxList(torch.zeros(0, 4), (0, 0), mode="xyxy")
            boxlist.add_field("objectness", 0)
            boxlist.add_field("scores", 0)
            boxlist.add_field("labels", -1)

        else:
            img_h, img_w = imgs.image_sizes[i]
            boxlist = BoxList(res[:, :4], (img_w, img_h), mode="xyxy")
            boxlist.add_field("objectness", res[:, 4])
            boxlist.add_field("scores", res[:, 5] * res[:, 4])
            boxlist.add_field("labels", res[:, 6] + 1)
        output[i] = boxlist

    return output



def reduce_mean(tensor):
    """"Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def xyxy2CxCywh(xyxy, size=None):
    x1 = xyxy[..., 0]
    y1 = xyxy[..., 1]
    x2 = xyxy[..., 2]
    y2 = xyxy[..., 3]

    cx = (x1+x2) /2
    cy = (y1+y2) /2

    w = x2 - x1
    h = y2 - y1
    if size is not None:
    	w = w.clamp(min=0, max=size[1])
    	h = h.clamp(min=0, max=size[0])
    return torch.stack([cx, cy, w, h], axis=-1)

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """

    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project', torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        shape = x.size()
        x = F.softmax(x.reshape(*shape[:-1], 4, self.reg_max + 1), dim=-1)
        b, nb, ne, _ = x.size()
        x = x.reshape(b*nb*ne, self.reg_max+1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, nb, 4)
        return x


class GFocalHeadV2(nn.Module):
    """Ref to Generalized Focal Loss V2: Learning Reliable Localization Quality
    Estimation for Dense Object Detection.
    """
    def __init__(self, num_classes, in_channels,
                 stacked_convs=4, feat_channels=96, reg_max=14, reg_topk=4, reg_channels=64,
                 strides=[8, 16, 32], add_mean=True, act='SiLU', start_kernel_size=3, conv_groups=2,
                 simOTA_cls_weight=1.0, simOTA_iou_weight=3.0, depth_mul=1.0, width_mul=1.0,
                 **kwargs):
        super(GFocalHeadV2, self).__init__()
        self.num_classes = num_classes
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
        self.strides = strides
        self.feat_channels = feat_channels if isinstance(feat_channels, list) else [feat_channels] * len(self.strides)
        self.feat_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))

        self.cls_out_channels = num_classes + 1 # add 1 for keep consistance with former models and will be deprecated in future.
        self.stacked_convs = stacked_convs
        self.conv_groups = conv_groups
        self.reg_max = reg_max
        self.reg_topk = reg_topk
        self.reg_channels = reg_channels
        self.add_mean = add_mean
        self.total_dim = reg_topk
        self.start_kernel_size = start_kernel_size
        self.decode_in_inference = True # will be set as False, when trying to convert onnx models

        self.act = act

        if add_mean:
            self.total_dim += 1

        self.assigner = SimOTAAssigner(center_radius=2.5, cls_weight=simOTA_cls_weight, iou_weight=simOTA_iou_weight)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)

        self._init_layers()
        self.init_weights()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        self.relu = nn.ReLU(inplace=True)
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = feat_channels if i > 0 else in_channel
            kernel_size = 3 if i > 0 else self.start_kernel_size
            cls_convs.append(ConvModule(chn, feat_channels, kernel_size, 1, kernel_size//2, groups=self.conv_groups, norm_cfg=dict(type='BN'), activation=self.act))
            reg_convs.append(ConvModule(chn, feat_channels, kernel_size, 1, kernel_size//2, groups=self.conv_groups, norm_cfg=dict(type='BN'), activation=self.act))

        conf_vector = [nn.Conv2d(4 * self.total_dim, self.reg_channels, 1)]
        conf_vector += [self.relu]
        conf_vector += [nn.Conv2d(self.reg_channels, 1, 1), nn.Sigmoid()]
        reg_conf = nn.Sequential(*conf_vector)

        return cls_convs, reg_convs, reg_conf

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.reg_confs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs, reg_conf = self._build_not_shared_convs(self.in_channels[i],self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)
            self.reg_confs.append(reg_conf)

        self.gfl_cls = nn.ModuleList(
                            [nn.Conv2d(self.feat_channels[i],self.cls_out_channels,
                                3, padding=1) for i in range(len(self.strides))])

        self.gfl_reg = nn.ModuleList(
                            [nn.Conv2d(self.feat_channels[i], 4 * (self.reg_max + 1),
                                3, padding=1) for i in range(len(self.strides))])

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conf in self.reg_confs:
            for m in reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = float(-np.log((1 - 0.01) / 0.01))
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)

        for m in self.modules():
            t = type(m)
            if t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True

    def forward(self, xin, gt_boxes=None, gt_labels=None, imgs=None, conf_thre=0.05, nms_thre=0.7):

        # prepare labels during training
        b, c, h, w = xin[0].shape
        '''
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field("labels") - 1).long()) # labels starts from 1
        '''
        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(
                xin[i].shape[0],
                xin[i].shape[-2:],
                stride,
                dtype=torch.float32,
                device=xin[0].device)
                for i, stride in enumerate(self.strides)]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)

        # forward for bboxes and classification prediction
        cls_scores, bbox_preds = multi_apply(
            self.forward_single,
            xin,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.reg_confs,
            self.scales,
            )
        flatten_cls_scores = torch.cat(cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(bbox_preds, dim=1)

        # calculating losses or bboxes decoded
        if self.training:
            loss = self.loss(
                flatten_cls_scores,
                flatten_bbox_preds,
                gt_boxes,
                gt_labels,
                mlvl_priors)
            return loss
        else:
            output = self.get_bboxes(
                flatten_cls_scores,
                flatten_bbox_preds,
                mlvl_priors)
            # if self.decode_in_inference:
            #    output = postprocess_gfocal(output, self.num_classes, conf_thre, nms_thre, imgs)
            return output

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, reg_conf, scale):
        """Forward feature of a single scale level.

        """
        cls_feat = x
        reg_feat = x

        for cls_conv in cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in reg_convs:
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), dim=2)
        prob_topk, _ = prob.topk(self.reg_topk, dim=2)

        if self.add_mean:
            stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                             dim=2)
        else:
            stat = prob_topk

        quality_score = reg_conf(stat.reshape(N, 4*self.total_dim, H, W))
        cls_score = gfl_cls(cls_feat).sigmoid() * quality_score

        flatten_cls_score = cls_score.flatten(start_dim=2).transpose(1, 2)
        flatten_bbox_pred = bbox_pred.flatten(start_dim=2).transpose(1, 2)
        return flatten_cls_score, flatten_bbox_pred

    def get_single_level_center_priors(self,
                                       batch_size,
                                       featmap_size,
                                       stride,
                                       dtype,
                                       device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype, device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype, device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             mlvl_center_priors,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        """
        device = cls_scores[0].device

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)
        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(torch.float).to(device)).item(), 1.0)

        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        loss_qfl = self.loss_cls(
            cls_scores, (labels, label_scores), avg_factor=num_total_pos)

        pos_inds = torch.nonzero(
                (labels >= 0) & (labels < self.num_classes), as_tuple=False).squeeze(1)

        temp_scores = bbox_overlaps(decoded_bboxes[pos_inds], bbox_targets[pos_inds], is_aligned=True)
        if len(pos_inds) > 0:
            weight_targets = cls_scores.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
                )
            loss_dfl = self.loss_dfl(
                bbox_preds[pos_inds].reshape(-1, self.reg_max+1),
                dfl_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * norm_factor,
                )

        else:
            loss_bbox = bbox_preds.sum() * 0.0
            loss_dfl = bbox_preds.sum() * 0.0

        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
                total_loss=total_loss,
                loss_cls=loss_qfl,
                loss_bbox=loss_bbox,
                loss_dfl=loss_dfl,
                )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
            all_bbox_weights, all_dfl_targets, all_pos_num)

    def get_target_single(self,
                           center_priors,
                           cls_scores,
                           bbox_preds,
                           gt_bboxes,
                           gt_labels,
                           unmap_outputs=True,
                           gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center, dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center, dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights,
                   bbox_targets, bbox_weights,  dfl_targets, 0)

        assign_result = self.assigner.assign(
            cls_scores.detach(),
            center_priors,
            bbox_preds.detach(),
            gt_bboxes,
            gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_bbox_targets, self.reg_max)
                / center_priors[pos_inds, None, 2]
            )
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets, bbox_weights,
                dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def get_bboxes(self, cls_preds, reg_preds, mlvl_center_priors, img_meta=None):

        device = cls_preds.device
        batch_size = cls_preds.shape[0]
        dis_preds = self.integral(reg_preds) * mlvl_center_priors[..., 2, None]
        bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)

        bboxes = xyxy2CxCywh(bboxes)
        obj = torch.ones_like(cls_preds[..., 0:1])
        res = torch.cat([bboxes, obj, cls_preds[..., 0:self.num_classes]], dim=-1)

        return res


if __name__ == "__main__":
    import torch
    in_channels = [96, 160, 384]
    model = GFocalHeadV2(80, in_channels)
    print(model)