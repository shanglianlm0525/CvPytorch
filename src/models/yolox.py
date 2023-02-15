# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 10:58
# @Author : liumin
# @File : yolox.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from ..losses import build_loss


def yolox_post_process(outputs, down_strides, num_classes, conf_thre, nms_thre):
    hw = [i.shape[-2:] for i in outputs]
    grids, strides = [], []
    for (hsize, wsize), stride in zip(hw, down_strides):
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides.append(torch.full((*shape, 1), stride))

    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)  # bs, all_anchor, 85(+128)
    grids = torch.cat(grids, dim=1).type(outputs.dtype).to(outputs.device)
    strides = torch.cat(strides, dim=1).type(outputs.dtype).to(outputs.device)

    # x, y
    outputs[..., 0:2] = (outputs[..., 0:2] + grids) * strides
    # w, h
    outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
    # obj
    outputs[..., 4:5] = torch.sigmoid(outputs[..., 4:5])
    # 80 class
    outputs[..., 5:5 + num_classes] = torch.sigmoid(outputs[..., 5:5 + num_classes])
    # reid
    reid_dim = outputs.shape[2] - num_classes - 5
    if reid_dim > 0:
        outputs[..., 5 + num_classes:] = F.normalize(outputs[..., 5 + num_classes:], dim=2)

    box_corner = outputs.new(outputs.shape)
    box_corner[:, :, 0] = outputs[:, :, 0] - outputs[:, :, 2] / 2  # x1
    box_corner[:, :, 1] = outputs[:, :, 1] - outputs[:, :, 3] / 2  # y1
    box_corner[:, :, 2] = outputs[:, :, 0] + outputs[:, :, 2] / 2  # x2
    box_corner[:, :, 3] = outputs[:, :, 1] + outputs[:, :, 3] / 2  # y2
    outputs[:, :, :4] = box_corner[:, :, :4]

    detections_list = []
    for i, image_pred in enumerate(outputs):
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            detections_list.append(None)
            continue
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5],
                                                    detections[:, 6], nms_thre)
        detections = detections[nms_out_index]
        detections_list.append(detections)
    return detections_list


class YOLOX(nn.Module):
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOX, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.depth_mul, self.width_mul = self.cfg[self.model_cfg.TYPE.split("_")[1]]
        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)

        self.loss = build_loss(self.model_cfg.LOSS)

        self.stride = [8, 16, 32]
        self.conf_thr = 0.01
        self.nms_thr = 0.65


    def setup_extra_params(self):
        self.model_cfg.BACKBONE.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.BACKBONE.__setitem__('width_mul', self.width_mul)
        self.model_cfg.NECK.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.NECK.__setitem__('width_mul', self.width_mul)
        self.model_cfg.HEAD.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.HEAD.__setitem__('width_mul', self.width_mul)
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)
        self.model_cfg.LOSS.__setitem__('num_classes', self.num_classes)


    def trans_specific_format(self, imgs, targets):
        max_labels = max([target['labels'].shape[0] for target in targets])
        new_gts = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for target in targets:
            new_gt = torch.zeros((max_labels, 5), device=imgs[0].device)
            gt_tmp = torch.cat([target['labels'].unsqueeze(1), target['boxes']], 1)
            new_gt[:gt_tmp.shape[0], :] = gt_tmp
            new_gts.append(new_gt)
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            if target.__contains__('height'):
                new_heights.append(target['height'])
            if target.__contains__('width'):
                new_widths.append(target['width'])

        t_targets = {}
        t_targets["gt"] = torch.stack(new_gts)
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets
        # return torch.stack(imgs), t_targets

    def forward(self, imgs, targets=None, mode='infer', **kwargs):

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)

            body_feats = self.backbone(imgs)
            neck_feats = self.neck(body_feats)
            out = self.head(neck_feats)

            losses = self.loss(out, targets["gt"])
            # print(losses)

            if mode == 'val':
                outputs = []
                predicts = yolox_post_process(out, self.stride, self.num_classes, self.conf_thr, self.nms_thr)

                for width, height, scale, pad, predict in zip(targets['width'], targets['height'], targets['scales'], targets['pads'], predicts):
                    if predict is not None:
                        # predict : x1, y1, x2, y2, obj_conf, class_conf, class_pred
                        bboxes_np = predict[:, :4].cpu().numpy()
                        width = width.cpu().numpy()
                        height = height.cpu().numpy()
                        scale = scale.cpu().numpy()
                        pad = pad.cpu().numpy()
                        bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                        bboxes_np[:, [1, 3]] -= pad[0]
                        bboxes_np[:, [0, 2]] /= scale[1]
                        bboxes_np[:, [1, 3]] /= scale[0]

                        # clip boxes
                        bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                        bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)
                        outputs.append(
                            {"boxes": torch.from_numpy(bboxes_np), "labels": predict[:, 6], "scores": predict[:, 4] * predict[:, 5]})
                    else:
                        outputs.append(
                            {"boxes": torch.empty((0, 4)), "labels": torch.empty((0, 1)), "scores": torch.empty((0, 1))})
                return losses, outputs
            else:
                return losses

