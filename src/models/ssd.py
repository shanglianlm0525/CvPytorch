# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/9/11 16:56
# @Author : liumin
# @File : ssd.py

import torch
import torch.nn as nn
from torchvision.ops import nms
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.resnet import resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from torchvision.models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from .anchors.prior_box import PriorBox
from ..losses.multibox_loss import MultiBoxLoss, decode


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not pretrained_path)
            self.out_channels = [256, 512, 512, 256, 256, 128]
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not pretrained_path)
            self.out_channels = [256, 512, 512, 256, 256, 256]
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not pretrained_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not pretrained_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not pretrained_path)
            self.out_channels = [1024, 512, 512, 256, 256, 256]
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class SSD(nn.Module):
    def __init__(self, dictionary=None):
        super().__init__()
        self.dictionary = dictionary
        self.input_size = [300, 300]

        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self._num_classes = len(self.dictionary) + 1
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.feature_extractor = ResNet('resnet50', None)
        self.add_extras(self.feature_extractor.out_channels)
        self.num_defaults = [4, 6, 6, 6, 4, 4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self._num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)

        self.softmax = nn.Softmax(dim=-1)

        self._priorbox = PriorBox()
        self._priors = self._priorbox.forward().cuda()

        self._criterion = MultiBoxLoss(self._num_classes, 0.5, True, 0, True, 3, 0.5, False)

        self.top_k = 200
        self.variance = [0.1, 0.2]
        self.conf_thresh = 0.01
        self.nms_thresh = 0.45

        self._init_weights()

    def add_extras(self, input_size):
        self.extras_layers = []
        for i, (input_size, output_size, channels) in enumerate(zip(input_size[:-1], input_size[1:], [256, 256, 128, 128, 128])):
            if i < 3:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, padding=1, stride=2, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(input_size, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels, output_size, kernel_size=3, bias=False),
                    nn.BatchNorm2d(output_size),
                    nn.ReLU(inplace=True),
                )

            self.extras_layers.append(layer)

        self.extras_layers = nn.ModuleList(self.extras_layers)

    def _init_weights(self):
        layers = [*self.extras_layers, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def multibox(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), -1, 4), c(s).view(s.size(0), -1, self._num_classes)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 1).contiguous(), torch.cat(confs, 1).contiguous()
        return locs, confs


    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        x = self.feature_extractor(imgs)

        detection_feed = [x]
        for l in self.extras_layers:
            x = l(x)
            detection_feed.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.multibox(detection_feed, self.loc, self.conf)

        if mode == 'infer':
            # locs: nbatch x 8732 x nlocs, confs: nbatch x 8732 x nlabels, self._priors: 8732 x nlocs
            out = self._detect(locs,self.softmax(confs),self._priors)

            return out
        else:
            # locs: nbatch x 8732 x nlocs, confs: nbatch x 8732 x nlabels, self._priors: 8732 x nlocs
            output = (locs,confs,self._priors)

            losses = {}
            losses['loc_loss'], losses['conf_loss'] = self._criterion(output, labels)
            losses['loss'] = losses['loc_loss'] + losses['conf_loss']

            if mode == 'val':
                performances = {}
                performances['performance'] = 1000-losses['loss']

                self.eval_voc(locs, self.softmax(confs), self._priors)

                return losses, performances
            else:
                return losses

    def eval_voc(self,loc_data,conf_data,prior_data):
        batch_size = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        output = torch.zeros(batch_size, self._num_classes, self.top_k, 5)
        conf_preds = conf_data.view(batch_size, num_priors, self._num_classes).transpose(2, 1)

        for i in range(batch_size):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self._num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                keep = nms(boxes, scores, self.nms_thresh)
                keep = keep[:self.top_k]
                count = keep.shape[0]
                output[i, cl, :count] = torch.cat((scores[keep[:count]].unsqueeze(1),boxes[keep[:count]]), 1)

        flt = output.contiguous().view(batch_size, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)

        # print('output',output.shape)
        return output