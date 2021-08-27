# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 10:58
# @Author : liumin
# @File : yolox.py


import torch
import torch.nn as nn
import torchvision

from .backbones import build_backbone
from .necks import build_neck
from .heads import build_head
from ..losses.yolox_loss import YoloxLoss
from ..utils.torch_utils import set_bn_momentum


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
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
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # _, conf_mask = torch.topk((image_pred[:, 4] * class_conf.squeeze()), 1000)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

class YOLOX(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOX, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.conf_thr = 0.01
        self.nms_thr = 0.65

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)

        self.loss = YoloxLoss(self.num_classes)

        self._init_weight()

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)

    def _init_weight(self):
        set_bn_momentum(self.backbone, momentum=0.03, eps=1e-3)
        set_bn_momentum(self.neck, momentum=0.03, eps=1e-3)
        set_bn_momentum(self.head, momentum=0.03, eps=1e-3)

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
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["gt"] = torch.stack(new_gts)
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return torch.stack(imgs), t_targets

    def forward(self, imgs, targets=None, mode='infer', **kwargs):

        if mode == 'infer':
            '''
                for inference mode, img should preprocessed before feeding in net 
            '''

            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)

            losses = {}
            origin_preds = []
            preds, grids, x_shifts, y_shifts, expanded_strides = self.head(self.neck(self.backbone(imgs)))
            losses['loss'], losses['iou_loss'],losses['conf_loss'],losses['cls_loss'],losses['l1_loss'] = self.loss(imgs, x_shifts, y_shifts, expanded_strides, targets['gt'], torch.cat(grids, 1), origin_preds, dtype=imgs[0].dtype)

            if mode == 'val':
                outputs = []
                self.hw = [x.shape[-2:] for x in preds]
                # [batch, n_anchors_all, 85]
                outs = torch.cat([x.flatten(start_dim=2) for x in preds], dim=2).permute(0, 2, 1)
                # dets torch.Size([64, 8400, 85])
                outs = self.decode_outputs(outs, dtype=imgs[0].type())
                dets = postprocess(outs, self.num_classes, self.conf_thr, self.nms_thr)

                for width, height, scale, pad, det in zip(targets['width'], targets['height'], targets['scales'], targets['pads'], dets):
                    # print('det', det)
                    if det is not None:
                        bboxes_np = det[:, :4].cpu().numpy()
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
                            {"boxes": torch.tensor(bboxes_np), "labels": det[:, 6], "scores": det[:, 4] * det[:, 5]})
                    else:
                        outputs.append(
                            {"boxes": torch.empty((0, 4)), "labels": torch.empty((0, 1)), "scores": torch.empty((0, 1))})
                return losses, outputs
            else:
                return losses


    def decode_outputs(self, outputs, dtype):
        strides1 = [8, 16, 32]
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, strides1):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
