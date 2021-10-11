# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/17 10:06
# @Author : liumin
# @File : openpose.py

import torch
import torch.nn as nn

from src.data.transforms.keypoint_target_transforms import get_openpose_ground_truth
from src.losses.openpose_loss import OpenPoseLoss
from src.models.backbones import build_backbone
from src.models.heads import build_head

'''
    OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
    https://arxiv.org/pdf/1812.08008.pdf
'''

class OpenPose(nn.Module):
    def __init__(self, dictionary=None, model_cfg=None):
        super(OpenPose, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 368, 368)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        self.backbone.layer3[-4] = nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.backbone.layer3[-2] = nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.head = build_head(self.model_cfg.HEAD)

        self.loss = OpenPoseLoss()
        # self.model_cfg.LOSS.num_classes = self.num_classes
        # self.loss = NanoDetLoss(self.model_cfg.LOSS)

    def setup_extra_params(self):
        self.model_cfg.HEAD.__setitem__('num_classes', 19)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def trans_specific_format(self, imgs, targets):
        new_heatmaps = []
        new_pafs = []
        new_scales = []
        new_pads = []
        new_heights = []
        new_widths = []
        for i, target in enumerate(targets):
            new_heatmaps.append(target['heatmaps'])
            new_pafs.append(target['pafs'])
            if target.__contains__('scales'):
                new_scales.append(target['scales'])
            if target.__contains__('pads'):
                new_pads.append(target['pads'])
            new_heights.append(target['height'])
            new_widths.append(target['width'])

        t_targets = {}
        t_targets["heatmaps"] = torch.stack(new_heatmaps, 0)
        t_targets["pafs"] = torch.stack(new_pafs, 0)
        t_targets["scales"] = new_scales if len(new_scales) > 0 else []
        t_targets["pads"] = new_pads if len(new_pads) > 0 else []
        t_targets["height"] = new_heights
        t_targets["width"] = new_widths
        return imgs, t_targets

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05
        if mode == 'infer':
            pass
        else:
            imgs, targets = self.trans_specific_format(imgs, targets)


            b, _, height, width = imgs.shape
            # imgs 16 x 3 x 368 x 368
            # targets [15.00000, 55.00000, 0.38317, 0.30502, 0.59623, 0.46391]

            losses = {}
            x = self.backbone(imgs)
            out, train_out = self.head(x)

            total_loss, losses = self.loss(train_out, targets["heatmaps"], targets["pafs"])
            losses['loss'] = total_loss
            # print(losses)


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
                return losses, outputs
            else:
                return losses