# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/5 18:38
# @Author : liumin
# @File : led_net.py

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import interpolate
import numpy as np

from src.losses.metric import SegmentationMetric

from CvPytorch.src.losses.metric import batch_pix_accuracy
from pytorch_segmentation.utils.metrics import batch_intersection_union


def split(x):
    c = int(x.size()[1])
    c1 = round(c * 0.5)
    x1 = x[:, :c1, :, :].contiguous()
    x2 = x[:, c1:, :, :].contiguous()

    return x1, x2


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class Conv2dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# after Concat -> BN, you also can use Dropout like SS_nbt_module may be make a good result!
class DownsamplerBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownsamplerBlock, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel - in_channel, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class SS_nbt_module(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        oup_inc = chann // 2

        # dw
        self.conv3x1_1_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_l = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_l = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_l = nn.BatchNorm2d(oup_inc, eps=1e-03)

        # dw
        self.conv3x1_1_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.conv3x1_2_r = nn.Conv2d(oup_inc, oup_inc, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                     dilation=(dilated, 1))

        self.conv1x3_2_r = nn.Conv2d(oup_inc, oup_inc, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                     dilation=(1, dilated))

        self.bn2_r = nn.BatchNorm2d(oup_inc, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def forward(self, input):
        # x1 = input[:,:(input.shape[1]//2),:,:]
        # x2 = input[:,(input.shape[1]//2):,:,:]
        residual = input
        x1, x2 = split(input)

        output1 = self.conv3x1_1_l(x1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_1_l(output1)
        output1 = self.bn1_l(output1)
        output1 = self.relu(output1)

        output1 = self.conv3x1_2_l(output1)
        output1 = self.relu(output1)
        output1 = self.conv1x3_2_l(output1)
        output1 = self.bn2_l(output1)

        output2 = self.conv1x3_1_r(x2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_1_r(output2)
        output2 = self.bn1_r(output2)
        output2 = self.relu(output2)

        output2 = self.conv1x3_2_r(output2)
        output2 = self.relu(output2)
        output2 = self.conv3x1_2_r(output2)
        output2 = self.bn2_r(output2)

        if (self.dropout.p != 0):
            output1 = self.dropout(output1)
            output2 = self.dropout(output2)

        out = self._concat(output1, output2)
        out = F.relu(residual + out, inplace=True)
        return channel_shuffle(out, 2)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 32)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(SS_nbt_module(32, 0.03, 1))

        self.layers.append(DownsamplerBlock(32, 64))

        for x in range(0, 2):
            self.layers.append(SS_nbt_module(64, 0.03, 1))

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 1))
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))

        for x in range(0, 1):
            self.layers.append(SS_nbt_module(128, 0.3, 2))
            self.layers.append(SS_nbt_module(128, 0.3, 5))
            self.layers.append(SS_nbt_module(128, 0.3, 9))
            self.layers.append(SS_nbt_module(128, 0.3, 17))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):

        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=True)
        return x


class APN_Module(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(APN_Module, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        # midddle branch
        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )
        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1)
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]

        b1 = self.branch1(x)
        # b1 = Interpolate(size=(h, w), mode="bilinear")(b1)
        b1 = interpolate(b1, size=(h, w), mode="bilinear", align_corners=True)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        # x3 = Interpolate(size=(h // 4, w // 4), mode="bilinear")(x3)
        x3 = interpolate(x3, size=(h // 4, w // 4), mode="bilinear", align_corners=True)
        x2 = self.conv2(x2)
        x = x2 + x3
        # x = Interpolate(size=(h // 2, w // 2), mode="bilinear")(x)
        x = interpolate(x, size=(h // 2, w // 2), mode="bilinear", align_corners=True)

        x1 = self.conv1(x1)
        x = x + x1
        # x = Interpolate(size=(h, w), mode="bilinear")(x)
        x = interpolate(x, size=(h, w), mode="bilinear", align_corners=True)

        x = torch.mul(x, mid)

        x = x + b1

        return x


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.apn = APN_Module(in_ch=128, out_ch=20)
        # self.upsample = Interpolate(size=(512, 1024), mode="bilinear")
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True)
        # self.output_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.apn(input)
        out = interpolate(output, size=(512, 1024), mode="bilinear", align_corners=True)
        # out = self.upsample(output)
        # print(out.shape)
        return out


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d,self).__init__()

        self.loss = nn.NLLLoss(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, dim=1), targets)

def computeIoU(x, y, nClasses,ignoreIndex):  # x=preds, y=targets
    # sizes should be "batch_size x nClasses x H x W"
    if (x.is_cuda or y.is_cuda):
        x = x.cuda()
        y = y.cuda()

    # if size is "batch_size x 1 x H x W" scatter to onehot
    if (x.size(1) == 1):
        x_onehot = torch.zeros(x.size(0), nClasses, x.size(2), x.size(3))
        if x.is_cuda:
            x_onehot = x_onehot.cuda()
        x_onehot.scatter_(1, x, 1).float()
    else:
        x_onehot = x.float()

    if (y.size(1) == 1):
        y_onehot = torch.zeros(y.size(0), nClasses, y.size(2), y.size(3))
        if y.is_cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, y, 1).float()
    else:
        y_onehot = y.float()

    if (ignoreIndex != -1):
        ignores = y_onehot[:, ignoreIndex].unsqueeze(1)
        x_onehot = x_onehot[:, :ignoreIndex]
        y_onehot = y_onehot[:, :ignoreIndex]
    else:
        ignores = 0

    tpmult = x_onehot * y_onehot  # times prediction and gt coincide is 1
    tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                   keepdim=True).squeeze()
    fpmult = x_onehot * (
                1 - y_onehot - ignores)  # times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
    fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                   keepdim=True).squeeze()
    fnmult = (1 - x_onehot) * (y_onehot)  # times prediction says its not that class and gt says it is
    fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3,
                   keepdim=True).squeeze()

    tp = tp.double().cpu()
    fp = fp.double().cpu()
    fn = fn.double().cpu()

    num = tp
    den = tp + fp + fn + 1e-15
    iou = num / den
    return torch.mean(iou)

# LEDNet
class LEDNet(nn.Module):
    def __init__(self, dictionary=None):
        super().__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 512, 1024)

        self._num_classes = len(self.dictionary) +1
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self.encoder = Encoder(self._num_classes)
        self.decoder = Decoder(self._num_classes)

        weight = torch.ones(self._num_classes)
        weight[0] = 2.3653597831726
        weight[1] = 4.4237880706787
        weight[2] = 2.9691488742828
        weight[3] = 5.3442072868347
        weight[4] = 5.2983593940735
        weight[5] = 5.2275490760803
        weight[6] = 5.4394111633301
        weight[7] = 5.3659925460815
        weight[8] = 3.4170460700989
        weight[9] = 5.2414722442627
        weight[10] = 4.7376127243042
        weight[11] = 5.2286224365234
        weight[12] = 5.455126285553
        weight[13] = 4.3019247055054
        weight[14] = 5.4264230728149
        weight[15] = 5.4331531524658
        weight[16] = 5.433765411377
        weight[17] = 5.4631009101868
        weight[18] = 5.3947434425354
        weight[19] = 0

        # criterion
        self._criterion = CrossEntropyLoss2d(weight.cuda()).cuda() # torch.nn.CrossEntropyLoss(weight=weight.cuda(), ignore_index=-1, reduction='mean').cuda()
        # evaluation metrics
        self._metric = SegmentationMetric(self._num_classes)

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

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        '''
        x = self.encoder(imgs)
        outputs = self.decoder.forward(x)
        '''
        outputs = self.encoder.forward(imgs, predict=True)

        if mode == 'infer':
            pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
            return pred
        else:
            device_id = labels.data.device
            dtype = imgs.data.dtype

            losses = {}

            losses['loss_crossEntropy'] = self._criterion(outputs, labels[:, 0])
            losses['loss'] = losses['loss_crossEntropy']
            if mode == 'val':
                performances = {}

                preds =  outputs.max(1)[1].unsqueeze(1).data
                labels = labels.data

                performances['performance_mIoU'] = computeIoU(preds, labels, self._num_classes, 19)
                performances['performance'] = performances['performance_mIoU']

                return losses, performances
            else:
                return losses

if __name__ == '__main__':
    img = torch.randn(2, 3, 512, 1024)
    model = LEDNet(18)
    # model.eval()
    outputs = model(img)
    print(outputs.shape)


