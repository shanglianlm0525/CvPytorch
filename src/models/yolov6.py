# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/26 10:58
# @Author : liumin
# @File : yolov6.py


import torch
import torch.nn as nn
import torchvision
import numpy as np
import time

from .backbones import build_backbone
from .detects import build_detect
from .modules.yolov6_modules import RepVGGBlock
from .necks import build_neck
from .heads import build_head
from ..losses import build_loss


def xywh2xyxy(x):
    # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates

    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output


class YOLOv6(nn.Module):
    cfg = {"n": [0.33, 0.25, 0.0, False],
            "t": [0.33, 0.375, 0.0, False],
            "s": [0.33, 0.5, 0.0, False],
            "m": [0.60, 0.75, float(2)/3, True],
            "l": [1.0, 1.0, float(1)/2, True]}
    def __init__(self, dictionary=None, model_cfg=None):
        super(YOLOv6, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.depth_mul, self.width_mul,self.csp_e, self.use_dfl = self.cfg[self.model_cfg.TYPE.split("_")[-1]]
        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.neck = build_neck(self.model_cfg.NECK)
        # self.head = build_head(self.model_cfg.HEAD)
        self.detect = build_detect(self.model_cfg.DETECT)

        self.loss = build_loss(self.model_cfg.LOSS)

        self.conf_thres = 0.03
        self.iou_thres = 0.65

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
        self.model_cfg.BACKBONE.__setitem__('csp_e', self.csp_e)
        self.model_cfg.NECK.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.NECK.__setitem__('width_mul', self.width_mul)
        self.model_cfg.NECK.__setitem__('csp_e', self.csp_e)
        self.model_cfg.DETECT.__setitem__('depth_mul', self.depth_mul)
        self.model_cfg.DETECT.__setitem__('width_mul', self.width_mul)
        self.model_cfg.DETECT.__setitem__('num_classes', self.num_classes)
        self.model_cfg.DETECT.__setitem__('use_dfl', self.use_dfl)
        self.model_cfg.LOSS.__setitem__('num_classes', self.num_classes)
        self.model_cfg.LOSS.__setitem__('use_dfl', self.use_dfl)

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

            body_feats = self.backbone(imgs)
            neck_feats = self.neck(body_feats)

            losses = {}
            if mode == 'val':
                out = self.detect(neck_feats)
                losses['loss'] = torch.tensor(0, device=out.device)

                outputs = []
                if out is not None:
                    preds = non_max_suppression(out, self.conf_thres, self.iou_thres, multi_label=True)  # N * 6
                    for i, (width, height, scale, pad, pred) in enumerate(
                            zip(targets['width'], targets['height'], targets['scales'], targets['pads'], preds)):
                        # print('pred:', pred)
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
                train_out = self.detect(neck_feats)
                losses['loss'], loss_states = self.loss(train_out, targets["gts"], 0)

                losses['iou_loss'] = loss_states[0]
                losses['dfl_loss'] = loss_states[1]
                losses['cls_loss'] = loss_states[2]
                return losses

