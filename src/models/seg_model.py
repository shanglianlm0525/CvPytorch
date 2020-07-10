# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/9 12:52
# @Author : liumin
# @File : seg_model.py

import torch
import torch.nn as nn
from PIL.Image import Image
from torchvision import models as modelsT
import numpy as np
import torch.nn.functional as torchF
from ..losses.dice_loss import dice_coeff

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        _, channel1, height1, width1 = x1.size()
        _, channel2, height2, width2 = x2.size()

        # input is CHW
        diffY = height2 - height1
        diffX = width2 - width1

        x1 = torchF.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SegModel(nn.Module):
    def __init__(self, dictionary=None):
        super(SegModel, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 600, 800)

        self._num_classes = len(self.dictionary)
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        bilinear = True
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, self._num_classes)

        self._criterion = nn.BCEWithLogitsLoss(weight=torch.from_numpy(np.array(self._weight)).float()).cuda()

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, img, labels=None, mode='infer', **kwargs):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        xx = self.up1(x5, x4)
        xx = self.up2(xx, x3)
        xx = self.up3(xx, x2)
        xx = self.up4(xx, x1)
        outputs = self.outc(xx)

        threshold = 0.5
        if mode == 'infer':
            mask = Image.fromarray(torch.clamp((torch.sigmoid(outputs) > threshold) * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy().astype(np.uint8))
            return mask
        else:
            losses = {}
            loss = self._criterion(outputs, labels)
            device_id = labels.data.device

            losses['all_loss'] = loss
            losses['all_loss_num'] = torch.as_tensor(labels.size(0), device=device_id)

            if mode == 'val':
                performances = {}
                outs = (outputs > threshold).float()
                performances['all_perf'] = torch.as_tensor(dice_coeff(outs, labels.squeeze(dim=1)).item(), device=device_id)
                performances['all_perf_num'] = torch.as_tensor(labels.size(0), device=device_id)

                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        if labels[idx].size(0):
                            losses[_label + '_loss'] = torchF.binary_cross_entropy_with_logits(outputs[:,idx,:,:], labels[:,idx,:,:]) * _weight
                            losses[_label + '_loss_num'] = torch.as_tensor(labels[:,idx,:,:].size(0), device=device_id)

                            performances[_label + '_perf'] = torch.as_tensor(dice_coeff(outs[:,idx,:,:], labels[:,idx,:,:].squeeze(dim=1)).item(),device=device_id)
                            performances[_label + '_perf_num'] = torch.as_tensor(labels[:,idx,:,:].size(0),device=device_id)
                        else:
                            losses[_label + '_loss'] = torch.as_tensor(0, device=device_id)
                            losses[_label + '_loss_num'] = torch.as_tensor(0, device=device_id)
                            performances[_label + '_perf'] = torch.as_tensor(0, device=device_id)
                            performances[_label + '_perf_num'] = torch.as_tensor(0, device=device_id)

                return losses, performances
            else:
                for idx, d in enumerate(self.dictionary):
                    for _label, _weight in d.items():
                        cognize = labels == idx
                        if labels[cognize].size(0):
                            losses[_label + '_loss'] = torchF.binary_cross_entropy_with_logits(outputs[cognize], labels[cognize]) * _weight
                            losses[_label + '_loss_num'] = torch.as_tensor(labels[cognize].size(0), device=device_id)
                        else:
                            losses[_label + '_loss'] = torch.as_tensor(0, device=device_id)
                            losses[_label + '_loss_num'] = torch.as_tensor(0, device=device_id)
                return losses


if __name__ =='__main__':
    model = SegModel()
    print(model)

    input = torch.randn(1,3,600,800)
    out = model(input)
    print(out.shape)
