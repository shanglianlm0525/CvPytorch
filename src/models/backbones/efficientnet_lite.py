# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/14 18:45
# @Author : liumin
# @File : efficientnet_lite.py

import math
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils import model_zoo

from src.models.modules.activations import act_layers


efficientnet_lite_params = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    'efficientnet_lite0': [1.0, 1.0, 224, 0.2],
    'efficientnet_lite1': [1.0, 1.1, 240, 0.2],
    'efficientnet_lite2': [1.1, 1.2, 260, 0.3],
    'efficientnet_lite3': [1.2, 1.4, 280, 0.3],
    'efficientnet_lite4': [1.4, 1.8, 300, 0.3],
}

model_urls = {
    'efficientnet_lite0': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite0.pth',
    'efficientnet_lite1': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite1.pth',
    'efficientnet_lite2': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite2.pth',
    'efficientnet_lite3': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite3.pth',
    'efficientnet_lite4': 'https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/efficientnet_lite4.pth',

}


def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False, activation='ReLU6'):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = act_layers(activation)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1 and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):

    def __init__(self, subtype='efficientnet_lite0', out_stages=[2,4,6], output_stride=32, backbone_path=None, pretrained = True):
        super(EfficientNetLite, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.backbone_path = backbone_path
        self.pretrained = pretrained


        width_multiplier, depth_multiplier, _, dropout_rate = efficientnet_lite_params[self.subtype]
        self.drop_connect_rate = 0.2
        self.out_stages = out_stages

        mb_block_settings = [
            # repeat|kernel_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],  # stage0
            [2, 3, 2, 6, 16, 24, 0.25],  # stage1 - 1/4
            [2, 5, 2, 6, 24, 40, 0.25],  # stage2 - 1/8
            [3, 3, 2, 6, 40, 80, 0.25],  # stage3
            [3, 5, 1, 6, 80, 112, 0.25],  # stage4 - 1/16
            [4, 5, 2, 6, 112, 192, 0.25],  # stage5
            [1, 3, 1, 6, 192, 320, 0.25]  # stage6 - 1/32
        ]

        self.out_channels = [16, 24, 40, 80, 112, 192, 320]
        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3),
            act_layers('ReLU6'),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            num_repeat, kernal_size, stride, expand_ratio, input_filters, output_filters, se_ratio = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = input_filters if i == 0 else round_filters(input_filters, width_multiplier)
            output_filters = round_filters(output_filters, width_multiplier)
            num_repeat = num_repeat if i == 0 or i == len(mb_block_settings) - 1 else round_repeats(num_repeat,
                                                                                                    depth_multiplier)

            # The first block needs to take care of stride and filter size increase.
            stage.append(
                MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio,
                                         has_se=False))

            self.blocks.append(stage)

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)


    def load_pretrained_weights(self):
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


    def forward(self, x):
        x = self.stem(x)
        output = []
        idx = 0
        for j, stage in enumerate(self.blocks):
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx += 1
            if j in self.out_stages:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]


if __name__=="__main__":
    model = EfficientNetLite('efficientnet_lite0')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)