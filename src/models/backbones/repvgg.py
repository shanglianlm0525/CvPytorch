# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/14 14:07
# @Author : liumin
# @File : repvgg.py

import torch
import torch.nn as nn
import numpy as np
from src.models.modules.activations import act_layers

"""
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
"""

optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {l: 2 for l in optional_groupwise_layers}
g4_map = {l: 4 for l in optional_groupwise_layers}

model_param = {
    'RepVGG_A0': dict(num_blocks=[2, 4, 14, 1], width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None),
    'RepVGG_A1': dict(num_blocks=[2, 4, 14, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None),
    'RepVGG_A2': dict(num_blocks=[2, 4, 14, 1], width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None),
    'RepVGG_B0': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[1, 1, 1, 2.5], override_groups_map=None),
    'RepVGG_B1': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=None),
    'RepVGG_B1g2': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map),
    'RepVGG_B1g4': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map),
    'RepVGG_B2': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None),
    'RepVGG_B2g2': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map,),
    'RepVGG_B2g4': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map),
    'RepVGG_B3': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=None),
    'RepVGG_B3g2': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map),
    'RepVGG_B3g4': dict(num_blocks=[4, 6, 16, 1], width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map),
}

class RepVGGConvModule(nn.Module):
    """
    RepVGG Conv Block from paper RepVGG: Making VGG-style ConvNets Great Again
    https://arxiv.org/abs/2101.03697
    https://github.com/DingXiaoH/RepVGG
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 activation='ReLU',
                 padding_mode='zeros',
                 deploy=False):
        super(RepVGGConvModule, self).__init__()
        assert activation is None or isinstance(activation, str)
        self.activation = activation

        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        # build activation layer
        if self.activation:
            self.act = act_layers(self.activation)

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels) if out_channels == in_channels and stride == 1 else None

            self.rbr_dense = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     groups=groups, bias=False),
                                           nn.BatchNorm2d(num_features=out_channels))

            self.rbr_1x1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                   kernel_size=1, stride=stride, padding=padding_11,
                                                   groups=groups, bias=False),
                                         nn.BatchNorm2d(num_features=out_channels))
            # print('RepVGG Block, identity = ', self.rbr_identity)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach().cpu().numpy(), bias.detach().cpu().numpy(),


class RepVGG(nn.Module):

    def __init__(self, subtype='RepVGG_A0', out_stages=[2,3,4], output_stride=32, backbone_path=None, pretrained = False):
        super(RepVGG, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        num_blocks = model_param[subtype]['num_blocks']
        width_multiplier = model_param[subtype]['width_multiplier']
        assert len(width_multiplier) == 4
        self.override_groups_map = model_param[subtype]['override_groups_map'] or dict()
        last_channel = 512
        self.deploy = False

        self.out_channels = [64, 64, 128, 256, 512]
        # width_multiplier: [0.75, 0.75, 0.75, 2.5]
        self.out_channels = [int(oc * width_multiplier[idx-1]) if idx>0 else min(64, int(64 * width_multiplier[0])) for idx, oc in enumerate(self.out_channels)]
        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = RepVGGConvModule(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                       activation='ReLU', deploy=self.deploy)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        out_planes = last_channel if last_channel else int(512 * width_multiplier[3])
        self.stage4 = self._make_stage(out_planes, num_blocks[3], stride=2)

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGConvModule(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                           stride=stride, padding=1, groups=cur_groups, activation='ReLU',
                                           deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0001)

    def forward(self, x):
        x = self.stage0(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output) if len(self.out_stages) > 1 else output[0]


if __name__=="__main__":
    model = RepVGG('RepVGG_A0')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)