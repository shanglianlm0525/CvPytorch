# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/14 10:45
# @Author : liumin
# @File : enet.py

"""
    ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
    https://arxiv.org/pdf/1606.02147.pdf
"""

import torch
import torch.nn as nn
import numpy as np

__all__ = ["ENet"]

from CvPytorch.src.losses.seg_loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d, DiceLoss, FocalLoss, \
    LovaszSoftmax, CE_DiceLoss


def Conv1x1BNReLU(in_channels,out_channels,is_relu=True):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if is_relu else nn.PReLU()
        )

def Conv2x2BNReLU(in_channels,out_channels,is_relu=True):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if is_relu else nn.PReLU()
        )

def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

def Conv3x3BNReLU(in_channels,out_channels,stride,dilation=1,is_relu=True):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=dilation,dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if is_relu else nn.PReLU()
        )

def TransposeConv3x3BNReLU(in_channels,out_channels,stride=2,is_relu=True):
    return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1,output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if is_relu else nn.PReLU()
        )


def AsymmetricConv(channels,stride,is_relu=False):
    return nn.Sequential(
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[5,1], stride=stride, padding=[2,0], bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True) if is_relu else nn.PReLU(),
        nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=[1,5], stride=stride, padding=[0,2], bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True) if is_relu else nn.PReLU()
    )

class InitialBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(InitialBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, kernel_size=3, stride=2,padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        return self.relu(self.bn(torch.cat([self.conv(x),self.pool(x)],dim=1)))


class RegularBottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1, expansion = 4,dilation=1,is_relu=False,asymmetric=False,p=0.01):
        super(RegularBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_places, mid_channels, False),
            AsymmetricConv(mid_channels, 1, is_relu) if asymmetric else Conv3x3BNReLU(mid_channels, mid_channels, 1,dilation, is_relu),
            Conv1x1BNReLU(mid_channels, places,is_relu),
            nn.Dropout2d(p=p)
        )
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)
        out += residual
        out = self.relu(out)
        return out


class DownBottleneck(nn.Module):
    def __init__(self,in_places,places, stride=2, expansion = 4,is_relu=False,p=0.01):
        super(DownBottleneck, self).__init__()
        mid_channels = in_places // expansion
        self.bottleneck = nn.Sequential(
            Conv2x2BNReLU(in_places, mid_channels, is_relu),
            Conv3x3BNReLU(mid_channels, mid_channels, 1, 1, is_relu),
            Conv1x1BNReLU(mid_channels, places,is_relu),
            nn.Dropout2d(p=p)
        )
        self.downsample = nn.MaxPool2d(3,stride=stride,padding=1,return_indices=True)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x):
        out = self.bottleneck(x)
        residual,indices = self.downsample(x)
        n, ch, h, w = out.size()
        ch_res = residual.size()[1]
        padding = torch.zeros(n, ch - ch_res, h, w)
        if residual.is_cuda:
            padding = padding.cuda()
        residual = torch.cat((residual, padding), 1)
        out += residual
        out = self.relu(out)
        return out, indices


class UpBottleneck(nn.Module):
    def __init__(self,in_places,places, stride=2, expansion = 4,is_relu=True,p=0.01):
        super(UpBottleneck, self).__init__()
        mid_channels = in_places // expansion

        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_places,mid_channels,is_relu),
            TransposeConv3x3BNReLU(mid_channels,mid_channels,stride,is_relu),
            Conv1x1BNReLU(mid_channels,places,is_relu),
            nn.Dropout2d(p=p)
        )
        self.upsample_conv = Conv1x1BN(in_places, places)
        self.upsample_unpool = nn.MaxUnpool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True) if is_relu else nn.PReLU()

    def forward(self, x, indices):
        out = self.bottleneck(x)
        residual = self.upsample_conv(x)
        residual = self.upsample_unpool(residual,indices)
        out += residual
        out = self.relu(out)
        return out


class ENet(nn.Module):
    def __init__(self, dictionary=None):
        super(ENet, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 480, 360)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.initialBlock = InitialBlock(3,16)
        self.stage1_1 = DownBottleneck(16, 64, 2, p=0.01)
        self.stage1_2 = nn.Sequential(
            RegularBottleneck(64, 64, 1, p=0.01),
            RegularBottleneck(64, 64, 1, p=0.01),
            RegularBottleneck(64, 64, 1, p=0.01),
            RegularBottleneck(64, 64, 1, p=0.01),
        )

        self.stage2_1 = DownBottleneck(64, 128, 2, p=0.1)
        self.stage2_2 = nn.Sequential(
            RegularBottleneck(128, 128, 1, p=0.1),
            RegularBottleneck(128, 128, 1, dilation=2, p=0.1),
            RegularBottleneck(128, 128, 1, asymmetric=True, p=0.1),
            RegularBottleneck(128, 128, 1, dilation=4, p=0.1),
            RegularBottleneck(128, 128, 1),
            RegularBottleneck(128, 128, 1, dilation=8, p=0.1),
            RegularBottleneck(128, 128, 1, asymmetric=True, p=0.1),
            RegularBottleneck(128, 128, 1, dilation=16, p=0.1),
        )
        self.stage3 = nn.Sequential(
            RegularBottleneck(128, 128, 1),
            RegularBottleneck(128, 128, 1, dilation=2, p=0.1),
            RegularBottleneck(128, 128, 1, asymmetric=True, p=0.1),
            RegularBottleneck(128, 128, 1, dilation=4, p=0.1),
            RegularBottleneck(128, 128, 1),
            RegularBottleneck(128, 128, 1, dilation=8, p=0.1),
            RegularBottleneck(128, 128, 1, asymmetric=True, p=0.1),
            RegularBottleneck(128, 128, 1, dilation=16, p=0.1),
        )
        self.stage4_1 = UpBottleneck(128, 64, 2, is_relu=True, p=0.1)
        self.stage4_2 = nn.Sequential(
            RegularBottleneck(64, 64, 1, is_relu=True, p=0.1),
            RegularBottleneck(64, 64, 1, is_relu=True, p=0.1),
        )
        self.stage5_1 = UpBottleneck(64, 16, 2, is_relu=True, p=0.1)
        self.stage5_2 = RegularBottleneck(16, 16, 1, is_relu=True, p=0.1)

        self.final_conv = nn.ConvTranspose2d(in_channels=16, out_channels=self._num_classes, kernel_size=3, stride=2, padding=1,
                           output_padding=1, bias=False)

        self.ce_criterion = CrossEntropyLoss2d(torch.from_numpy(np.array(self._weight)).float()).cuda()
        self.focal_criterion = FocalLoss(alpha=torch.from_numpy(np.array(self._weight)).float()).cuda()
        self.lovasz_criterion = LovaszSoftmax().cuda()
        self.bce_criterion = BCEWithLogitsLoss2d(pos_weight=torch.from_numpy(np.array(self._weight)).float()).cuda()
        self.dice_criterion = DiceLoss().cuda()
        self.ce_dice_criterion = CE_DiceLoss(weight=torch.from_numpy(np.array(self._weight)).float()).cuda()
        self.ce_criterion1 = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(self._weight)).float()).cuda()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        x = self.initialBlock(imgs)
        x,indices1 = self.stage1_1(x)
        x = self.stage1_2(x)
        x, indices2 = self.stage2_1(x)
        x = self.stage2_2(x)
        x = self.stage3(x)
        x = self.stage4_1(x, indices2)
        x = self.stage4_2(x)
        x = self.stage5_1(x, indices1)
        x = self.stage5_2(x)
        outputs = self.final_conv(x)

        if mode == 'infer':
            pass
        else:
            losses = {}
            losses['ce_loss'] = self.ce_criterion(outputs, targets)
            losses['focal_loss'] = self.focal_criterion(outputs, targets)
            losses['lovasz_loss'] = self.lovasz_criterion(outputs, targets)

            losses['bce_loss'] = self.bce_criterion(outputs, targets)
            losses['dice_loss'] = self.dice_criterion(outputs, targets)
            losses['ce_dice_loss'] = self.ce_dice_criterion(outputs, targets)
            losses['ce1_loss'] = self.ce_criterion1(outputs, targets)
            losses['loss'] = losses['ce_loss']
            print(losses)

            if mode == 'val':
                return losses, torch.argmax(outputs, dim=1).unsqueeze(1)
            else:
                return losses
