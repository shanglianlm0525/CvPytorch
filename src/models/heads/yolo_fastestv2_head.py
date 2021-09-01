# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/23 15:48
# @Author : liumin
# @File : yolo_fastestv2_head.py


import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class YOLOFastestv2Head(nn.Module):
    def __init__(self, num_classes=80, input_channel=72, stride=[16., 32.], anchors=[[12.64,19.39, 37.88,51.48, 55.71,138.31], [126.91,78.23, 131.57,214.55, 279.92,258.87]], device = 'cuda:0'):
        super(YOLOFastestv2Head, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_anchors = len(anchors[0]) // 2  # number of anchors
        self.stride = stride
        self.anchors = torch.from_numpy(np.array(anchors).reshape(-1, self.num_anchors, 2)).to(device)
        self.device = device

        self.output_reg_layers = nn.Conv2d(input_channel, 4 * self.num_anchors, 1, 1, 0, bias=True)
        self.output_obj_layers = nn.Conv2d(input_channel, self.num_anchors, 1, 1, 0, bias=True)
        self.output_cls_layers = nn.Conv2d(input_channel, self.num_classes, 1, 1, 0, bias=True)

    def forward(self, x):
        reg_outs, obj_outs, cls_outs = x
        train_out = self.output_reg_layers(reg_outs[0]), self.output_obj_layers(obj_outs[0]), self.output_cls_layers(cls_outs[0]), \
               self.output_reg_layers(reg_outs[1]), self.output_obj_layers(obj_outs[1]), self.output_cls_layers(cls_outs[1])

        out = self.handle_preds(train_out)
        return out, train_out

    def handle_preds(self, preds):
        output_bboxes = []
        for i in range(len(preds) // 3):
            bacth_bboxes = []
            reg_preds = preds[i * 3]
            obj_preds = preds[(i * 3) + 1]
            cls_preds = preds[(i * 3) + 2]

            for r, o, c in zip(reg_preds, obj_preds, cls_preds):
                r = r.permute(1, 2, 0)
                r = r.reshape(r.shape[0], r.shape[1], self.num_anchors, -1)

                o = o.permute(1, 2, 0)
                o = o.reshape(o.shape[0], o.shape[1], self.num_anchors, -1)

                c = c.permute(1, 2, 0)
                c = c.reshape(c.shape[0], c.shape[1], 1, c.shape[2])
                c = c.repeat(1, 1, 3, 1)

                anchor_boxes = torch.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

                # 计算anchor box的cx, cy
                grid = self.make_grid(r.shape[0], r.shape[1]).to(r.device)
                stride = self.stride[i]
                anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride

                # 计算anchor box的w, h
                anchors_cfg = self.anchors[i]
                anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg  # wh

                # 计算obj分数
                anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()

                # 计算cls分数
                anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim=3)

                # torch tensor 转为 numpy array
                anchor_boxes = anchor_boxes.cpu().detach().numpy()
                bacth_bboxes.append(anchor_boxes)

                # n, anchor num, h, w, box => n, (anchor num*h*w), box
            bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
            bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1])
            output_bboxes.append(bacth_bboxes)
        # merge
        return torch.cat(output_bboxes, 1)


    def make_grid(self, h, w):
        hv, wv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        return torch.stack((wv, hv), 2).repeat(1, 1, 3).reshape(h, w, self.num_anchors, -1)
