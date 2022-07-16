# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/4 17:03
# @Author : liumin
# @File : fastestdet.py

import torch
import torch.nn as nn
import torchvision

from src.losses import build_loss
from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.models.necks import build_neck


# 后处理(归一化后的坐标)
def non_max_suppression(preds, conf_thresh=0.25, nms_thresh=0.35):
    total_bboxes, output_bboxes = [], []
    # 将特征图转换为检测框的坐标
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    # 前背景分类分支
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # 检测框回归分支
    preg = pred[:, :, :, 1:5]
    # 目标类别分类分支
    pcls = pred[:, :, :, 5:]

    # 检测框置信度
    bboxes[..., 4] = pobj.squeeze(-1) * pcls.max(dim=-1)[0]
    bboxes[..., 5] = pcls.argmax(dim=-1)

    # 检测框的坐标
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)])
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid()
    bcx = (preg[..., 0].tanh() + gx.cuda()) / W
    bcy = (preg[..., 1].tanh() + gy.cuda()) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H * W, 6)
    total_bboxes.append(bboxes)

    batch_bboxes = torch.cat(total_bboxes, 1)

    # 对检测框进行NMS处理
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # 阈值筛选
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).cuda()
            c = torch.Tensor(c).squeeze(1).cuda()
            s = torch.Tensor(s).squeeze(1).cuda()
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes


class FastestDet(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(FastestDet, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 512, 512)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        self.head = build_head(self.model_cfg.HEAD)

        self.loss = build_loss(self.model_cfg.LOSS)

        self.conf_thres = 0.001
        self.nms_thres = 0.35


    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', self.num_classes)

    def trans_specific_format(self, imgs, targets):
        new_gts = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for i, target in enumerate(targets):
            new_gt = torch.zeros((target['labels'].shape[0], 6), device=target['labels'].device)
            new_gt[:, 0] = i
            new_gt[:, 1:] = torch.cat([target['labels'].unsqueeze(1), target['boxes']], 1)
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
        t_targets["gts"] = torch.cat(new_gts, 0)
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
            B, _, H, W = imgs.shape

            body_feats = self.backbone(imgs)
            neck_feats = self.neck(body_feats)
            out = self.head(neck_feats)

            losses = {}
            if out is not None:
                loss_states = self.loss(out, targets["gts"])
                losses['iou_loss'] = loss_states[0]
                losses['obj_loss'] = loss_states[1]
                losses['cls_loss'] = loss_states[2]
                losses['loss'] = loss_states[3]
            else:
                losses['loss'] = torch.tensor(0, device=imgs.device)

            if mode == 'val':
                outputs = []
                if out is not None:
                    preds = non_max_suppression(out, self.conf_thres, self.nms_thres)
                    for i, (width, height, scale, pad, pred) in enumerate(
                            zip(targets['width'], targets['height'], targets['scales'], targets['pads'], preds)):
                        # print('pred', pred)
                        scale = scale.cpu().numpy()
                        pad = pad.cpu().numpy()
                        width = width.cpu().numpy()
                        height = height.cpu().numpy()
                        predn = pred.clone()

                        bboxes_np = predn[:, :4].cpu().numpy()
                        bboxes_np[:, [0, 2]] *= W
                        bboxes_np[:, [1, 3]] *= H
                        bboxes_np[:, [0, 2]] -= pad[1]  # x padding
                        bboxes_np[:, [1, 3]] -= pad[0]
                        bboxes_np[:, [0, 2]] /= scale[1]
                        bboxes_np[:, [1, 3]] /= scale[0]

                        # clip boxes
                        bboxes_np[:, [0, 2]] = bboxes_np[:, [0, 2]].clip(0, width)
                        bboxes_np[:, [1, 3]] = bboxes_np[:, [1, 3]].clip(0, height)
                        outputs.append({"boxes": torch.tensor(bboxes_np), "labels": pred[:, 5], "scores": pred[:, 4]})
                        # outputs.append({"boxes": pred[:, :4], "labels": pred[:, 5], "scores": pred[:, 4]})
                return losses, outputs
            else:
                return losses
