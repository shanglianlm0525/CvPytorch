# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:02
# @Author : liumin
# @File : deeplabv3_plus.py

"""
    Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
    https://arxiv.org/pdf/1802.02611.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

from src.models.modules.aspp import ASPP
from src.evaluation.eval_segmentation import SegmentationEvaluator


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained_path=None):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=not pretrained_path)
            self.final_out_channels = 256
            self.low_level_inplanes = 64
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=not pretrained_path)
            self.final_out_channels = 1024
            self.low_level_inplanes = 256
        if pretrained_path:
            backbone.load_state_dict(torch.load(pretrained_path))


        self.early_extractor = nn.Sequential(*list(backbone.children())[:5])
        self.later_extractor = nn.Sequential(*list(backbone.children())[5:7])

        conv4_block1 = self.later_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.early_extractor(x)
        out = self.later_extractor(x)
        return out,x


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes=256):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Deeplabv3Plus(nn.Module):
    def __init__(self, dictionary=None):
        super().__init__()
        self.dictionary = dictionary
        self.input_size = [512, 512]
        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.backbone = ResNet('resnet50', None)
        self.aspp = ASPP(inplanes=self.backbone.final_out_channels)
        self.decoder = Decoder(self._num_classes, self.backbone.low_level_inplanes)

        self.bce_criterion = nn.BCEWithLogitsLoss().cuda()
        self._evaluator = SegmentationEvaluator(self._num_classes)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        x, low_level_feat = self.backbone(imgs)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        outputs = F.interpolate(x, size=imgs.size()[2:], mode='bilinear', align_corners=True)

        if mode == 'infer':

            return outputs
        else:
            losses = {}
            losses['bce_loss'] = 0

            for idx, d in enumerate(self.dictionary):
                for _label, _weight in d.items():
                    targets_onehot = torch.zeros_like(targets)
                    targets_onehot[targets == idx] = 1
                    losses['bce_loss'] += self.bce_criterion(outputs[:, idx, :, :].unsqueeze(dim=1), targets_onehot.float()) * _weight
            losses['loss'] = losses['bce_loss']

            if mode == 'val':
                return losses, torch.argmax(outputs, dim=1).unsqueeze(1)
            else:
                return losses

