# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/17 13:05
# @Author : liumin
# @File : unet.py

"""
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    https://arxiv.org/pdf/1505.04597.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..losses.dice_loss import dice_coeff


def Conv3x3BNReLU(in_channels,out_channels,stride,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3x3BNReLU(in_channels, out_channels,stride=1),
            Conv3x3BNReLU(out_channels, out_channels, stride=1)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=stride)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.pool(self.double_conv(x))


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels,bilinear=True):
        super().__init__()
        self.reduce = Conv1x1BNReLU(in_channels, in_channels//2)
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(self.reduce(x1))
        _, channel1, height1, width1 = x1.size()
        _, channel2, height2, width2 = x2.size()

        # input is CHW
        diffY = height2 - height1
        diffX = width2 - width1

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, dictionary=None):
        super(UNet, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 800, 600)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.conv = DoubleConv(3, 64)
        self.down1 = DownConv(64, 128)
        self.down2 = DownConv(128, 256)
        self.down3 = DownConv(256, 512)
        self.down4 = DownConv(512, 1024)
        self.up1 = UpConv(1024, 512)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)
        self.outconv = nn.Conv2d(64, self._num_classes, kernel_size=1)

        self.bce_criterion = nn.BCEWithLogitsLoss().cuda()

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
        x1 = self.conv(imgs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)
        outputs = self.outconv(xx)

        threshold = 0.5
        if mode == 'infer':
            probs = torch.sigmoid(outputs)
            probs[probs < threshold] = 0
            return probs * 255
        else:
            losses = {}
            device_id = targets.data.device

            losses['loss'] = 0
            losses['bce_loss'] = 0

            for idx, d in enumerate(self.dictionary):
                for _label, _weight in d.items():
                    targets_onehot = torch.zeros_like(targets)
                    targets_onehot[targets == idx] = 1
                    losses['bce_loss'] += self.bce_criterion(outputs[:, idx, :, :].unsqueeze(1), targets_onehot.float()) * _weight

            losses['loss'] = losses['bce_loss']

            if mode == 'val':
                '''
                probs = torch.sigmoid(outputs)
                outs = (probs > threshold).float()
                performances = {}
                performances['performance'] = 0

                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        targets_onehot = torch.zeros_like(targets)
                        targets_onehot[targets == idx] = 1
                        performances[_label + '_performance'] = torch.as_tensor(
                            dice_coeff(outs[:, idx, :, :].unsqueeze(1), targets_onehot).item(),device=device_id)
                        performances['performance'] += performances[_label + '_performance']

                performances['performance'] = performances['performance'] / len(self.dictionary)
                '''
                return losses, torch.argmax(outputs, dim=1).unsqueeze(1)
            else:
                return losses