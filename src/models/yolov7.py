# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/7 19:09
# @Author : liumin
# @File : yolov7.py
import time

import torch
import torch.nn as nn
import numpy as np
import torchvision

from src.models.backbones import build_backbone
from src.models.necks import build_neck
from src.models.heads import build_head
from src.models.detects import build_detect
from src.losses import build_loss

"""
    YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors
    https://arxiv.org/pdf/2207.02696.pdf
    https://github.com/WongKinYiu/yolov7
"""

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output



class YOLOv7(nn.Module):
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    '''
    anchors = [[[ 1.50000,  2.00000], [ 2.37500,  4.50000], [ 5.00000,  3.50000]],
        [[ 2.25000,  4.68750],[ 4.75000,  3.43750], [ 4.50000,  9.12500]],
        [[ 4.43750,  3.43750], [ 6.00000,  7.59375], [14.34375, 12.53125]]]
    '''
    anchors = [[[12, 16], [19, 36], [40, 28]],
               [[36, 75], [76, 55], [72, 146]],
               [[142, 110], [192, 243], [459, 401]]]
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOv7, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.depth_mul, self.width_mul = self.cfg[self.model_cfg.TYPE.split("_")[1]]
        self.setup_extra_params()
        # backbone
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        # neck
        self.neck = build_neck(self.model_cfg.NECK)
        # head
        self.head = build_head(self.model_cfg.HEAD)
        # detect
        self.detect = build_detect(self.model_cfg.DETECT)
        # loss
        self.loss = build_loss(self.model_cfg.LOSS)

        self.conf_thres = 0.001  # confidence threshold
        self.iou_thres = 0.6  # NMS IoU threshold

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True

    def setup_extra_params(self):
        self.model_cfg.BACKBONE.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.BACKBONE.__setitem__('width_mul', self.width_mul)
        self.model_cfg.NECK.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.NECK.__setitem__('width_mul', self.width_mul)
        self.model_cfg.HEAD.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.HEAD.__setitem__('width_mul', self.width_mul)
        self.model_cfg.DETECT.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.DETECT.__setitem__('width_mul', self.width_mul)
        self.model_cfg.DETECT.__setitem__('anchors', self.anchors)
        self.model_cfg.DETECT.__setitem__('num_classes', self.num_classes)
        self.model_cfg.LOSS.__setitem__('anchors', self.anchors)
        self.model_cfg.LOSS.__setitem__('num_classes', self.num_classes)


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
            ''' for inference mode, img should preprocessed before feeding in net '''
            return
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)
            b, _, height, width = imgs.shape
            # imgs N x 3 x 640 x 640
            losses = {}
            out, train_out = self.detect(self.head(self.neck(self.backbone(imgs))))
            # print(out, train_out)
            if train_out is not None:
                losses['loss'], loss_states = self.loss(train_out, targets["gts"], imgs)

                losses['box_loss'] = loss_states[0]
                losses['obj_loss'] = loss_states[1]
                losses['cls_loss'] = loss_states[2]
            else:
                losses['loss'] = torch.tensor(0, device=out.device)

            if mode == 'val':
                outputs = []
                if out is not None:
                    preds = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True)  # N * 6
                    for i, (width, height, scale, pad, pred) in enumerate(
                            zip(targets['width'], targets['height'], targets['scales'], targets['pads'], preds)):
                        scale = scale.cpu().numpy()
                        pad = pad.cpu().numpy()
                        width = width.cpu().numpy()
                        height = height.cpu().numpy()
                        predn = pred.clone()

                        bboxes_np = predn[:, :4].cpu().numpy()
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
