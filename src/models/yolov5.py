# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/12/13 19:14
# @Author : liumin
# @File : yolov5.py


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import nms

from src.losses import Yolov5Loss
from src.models.modules.yolov5_modules import Conv, C3, C3TR, SPPF, SPP, CBAM, Detect

"""
    https://github.com/ultralytics/yolov5
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
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
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
        i = nms(boxes, scores, iou_thres)  # NMS
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


class YOLOv5(nn.Module):
    anchors = [[[ 1.25000,  1.62500], [ 2.00000,  3.75000], [ 4.12500,  2.87500]],
        [[ 1.87500,  3.81250],[ 3.87500,  2.81250], [ 3.68750,  7.43750]],
        [[ 3.62500,  2.81250], [ 4.87500,  6.18750], [11.65625, 10.18750]]]
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOv5, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        if self.model_cfg.TYPE == 'yolov5s':
            self.depth_mul = 0.33  # model depth multiple
            self.width_mul = 0.50  # layer channel multiple
        elif self.model_cfg.TYPE == 'yolov5m':
            self.depth_mul = 0.67  # model depth multiple
            self.width_mul = 0.75  # layer channel multiple
        elif self.model_cfg.TYPE == 'yolov5l':
            self.depth_mul = 1.0  # model depth multiple
            self.width_mul = 1.0  # layer channel multiple
        elif self.model_cfg.TYPE == 'yolov5x':
            self.depth_mul = 1.33  # model depth multiple
            self.width_mul = 1.25  # layer channel multiple
        else:
            raise NotImplementedError

        stride = [8., 16., 32.]
        layers = [3, 6, 9, 3, 3, 3, 3, 3]
        out_channels = [64, 128, 256, 512, 1024]
        in_places = list(map(lambda x: int(x * self.width_mul), out_channels))
        layers = list(map(lambda x: max(round(x * self.depth_mul), 1), layers))

        self.setup_extra_params()
        self.backbone_conv1 = Conv(3, in_places[0], k=(6, 6), s=(2, 2), p=(2, 2))  # 0
        self.backbone_layer1 = nn.Sequential(Conv(in_places[0], in_places[1], 3, 2),
                                             C3(in_places[1], in_places[1], layers[0]))  # 2
        self.backbone_layer2 = nn.Sequential(Conv(in_places[1], in_places[2], 3, 2),
                                             C3(in_places[2], in_places[2], layers[1]))
        self.backbone_layer3 = nn.Sequential(Conv(in_places[2], in_places[3], 3, 2),  # 4
                                             C3(in_places[3], in_places[3], layers[2]))
        self.backbone_layer4 = nn.Sequential(Conv(in_places[3], in_places[4], 3, 2),  # 6
                                             C3(in_places[4], in_places[4], layers[3]),
                                             SPPF(in_places[4], in_places[4], 5))  # 9

        self.head_up_1_conv = Conv(in_places[4], in_places[3], 1, 1)  # 10
        self.head_up_1 = C3(in_places[3] * 2, in_places[3], layers[4], False)  # 13

        self.head_up_2_conv = Conv(in_places[3], in_places[2], 1, 1)  # 14
        self.head_up_2 = C3(in_places[2] * 2, in_places[2], layers[5], False)  # 17

        self.head_down_1_conv = Conv(in_places[2], in_places[2], 3, 2)       # 18
        self.head_down_1 = C3(in_places[3], in_places[3], layers[6], False)  # 20

        self.head_down_2_conv = Conv(in_places[3], in_places[3], 3, 2)         # 21
        self.head_down_2 = C3(in_places[4], in_places[4], layers[7], False)  # 33

        # nc=80, stride=[4.,  8., 16., 32.], anchors=(), ch=()
        self.detect = Detect(num_classes=self.num_classes, stride=stride, anchors=self.anchors, ch=in_places[2:])

        self.loss = Yolov5Loss(self.num_classes, anchors=self.anchors)

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
        pass

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
            # imgs 2 x 3 x 640 x 640
            # targets [15.00000, 55.00000, 0.38317, 0.30502, 0.59623, 0.46391]
            losses = {}
            out2 = self.backbone_layer1(self.backbone_conv1(imgs))
            out4 = self.backbone_layer2(out2)
            out6 = self.backbone_layer3(out4)
            out9 = self.backbone_layer4(out6)

            # up
            out10 = self.head_up_1_conv(out9)
            out13 = self.head_up_1(torch.cat([F.interpolate(out10, scale_factor=2, mode="nearest"), out6], dim=1))
            out14 = self.head_up_2_conv(out13)
            out17 = self.head_up_2(torch.cat([F.interpolate(out14, scale_factor=2, mode="nearest"), out4], dim=1))

            # down
            out20 = self.head_down_1(torch.cat([self.head_down_1_conv(out17), out14], dim=1))
            out23 = self.head_down_2(torch.cat([self.head_down_2_conv(out20), out10], dim=1))

            out, train_out = self.detect([out17, out20, out23])
            losses['loss'], loss_states = self.loss(train_out, targets["gts"])

            losses['box_loss'] = loss_states[0]
            losses['obj_loss'] = loss_states[1]
            losses['cls_loss'] = loss_states[2]

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
