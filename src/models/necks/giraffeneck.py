# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/7/19 10:39
# @Author : liumin
# @File : giraffeneck.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.shufflenetv2 import InvertedResidual

from src.models.modules.convs import ConvModule, DepthwiseConvModule


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="SiLU",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, 1, 0, norm_cfg=dict(type='BN'), activation='SiLU')
        self.conv2 = ConvModule(hidden_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='SiLU')
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvModule(in_channels, hidden_channels, kernel_size=1, norm_cfg=dict(type='BN'), activation='SiLU')
        self.conv2 = ConvModule(in_channels, hidden_channels, kernel_size=1, norm_cfg=dict(type='BN'), activation='SiLU')
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, kernel_size=1, norm_cfg=dict(type='BN'), activation='SiLU')
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class ResampleFeatureMap(nn.Sequential):
    def __init__(self, in_channels, out_channels, reduction_ratio=1.):
        super(ResampleFeatureMap, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio

        conv = None
        if in_channels != out_channels:
            conv = ConvModule(in_channels, out_channels, kernel_size=1, norm_cfg=dict(type='BN'), activation='Swish')

        if reduction_ratio > 1:
            if conv is not None:
                self.add_module('conv', conv)
            self.add_module('downsample', nn.MaxPool2d(kernel_size=3, stride=int(reduction_ratio), padding=1))
        else:
            if conv is not None:
                self.add_module('conv', conv)
            if reduction_ratio < 1:
                self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=int(1 // reduction_ratio)))


class GiraffeCombine(nn.Module):
    def __init__(self, in_channels, stride, fpn_config, fpn_channels, inputs_offsets, target_reduction, weight_method='attn'):
        super(GiraffeCombine, self).__init__()
        self.in_channels = in_channels
        self.stride = stride
        self.inputs_offsets = inputs_offsets
        self.weight_method = weight_method

        self.resample = nn.ModuleDict()
        reduction_base = stride[0]

        target_channels_idx = int(math.log(target_reduction // reduction_base, 2))
        for idx, offset in enumerate(inputs_offsets):
            if offset < len(in_channels):
                in_channel = in_channels[offset]
                input_reduction = stride[offset]
            else:
                node_idx = offset
                input_reduction = fpn_config[node_idx]['reduction']
                # in_channels = fpn_config[node_idx]['num_chs']
                input_channels_idx = int(math.log(input_reduction // reduction_base, 2))
                in_channel = in_channels[input_channels_idx]

            reduction_ratio = target_reduction / input_reduction
            if weight_method == 'concat':
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channel, in_channel, reduction_ratio=reduction_ratio)
            else:
                self.resample[str(offset)] = ResampleFeatureMap(
                    in_channel, fpn_channels[target_channels_idx], reduction_ratio=reduction_ratio)

        if weight_method == 'concat':
            src_channels = fpn_channels[target_channels_idx] * len(inputs_offsets)
            target_channels = fpn_channels[target_channels_idx]

        if weight_method == 'attn' or weight_method == 'fastattn':
            self.edge_weights = nn.Parameter(torch.ones(len(inputs_offsets)), requires_grad=True)  # WSM
        else:
            self.edge_weights = None

    def forward(self, x):
        dtype = x[0].dtype
        nodes = []
        if len(self.inputs_offsets) == 0:
            return None
        for offset, resample in zip(self.inputs_offsets, self.resample.values()):
            input_node = x[offset]
            input_node = resample(input_node)
            nodes.append(input_node)

        if self.weight_method == 'attn':
            normalized_weights = torch.softmax(self.edge_weights.to(dtype=dtype), dim=0)
            out = torch.stack(nodes, dim=-1) * normalized_weights
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'fastattn':
            edge_weights = nn.functional.relu(self.edge_weights.to(dtype=dtype))
            weights_sum = torch.sum(edge_weights)
            out = torch.stack(
                [(nodes[i] * edge_weights[i]) / (weights_sum + 0.0001) for i in range(len(nodes))], dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'sum':
            out = torch.stack(nodes, dim=-1)
            out = torch.sum(out, dim=-1)
        elif self.weight_method == 'concat':
            out = torch.cat(nodes, dim=1)
        else:
            raise ValueError('unknown weight_method {}'.format(self.weight_method))
        return out


class GiraffeNode(nn.Module):
    """ A simple wrapper used in place of nn.Sequential for torchscript typing
    Handles input type List[Tensor] -> output type Tensor
    """
    def __init__(self, combine: nn.Module, after_combine: nn.Module):
        super(GiraffeNode, self).__init__()
        self.combine = combine
        self.after_combine = after_combine

    def forward(self, x):
        combine_feat = self.combine(x)
        if combine_feat is None:
            return None
        else:
            return self.after_combine(combine_feat)

class GiraffeLayer(nn.Module):
    def __init__(self, in_channels, strides, fpn_config, inner_fpn_channels, outer_fpn_channels,
                 separable_conv=False, merge_type='conv'):
        super(GiraffeLayer, self).__init__()
        self.in_channels = in_channels
        self.strides = strides
        self.num_levels = len(in_channels)
        self.conv_bn_relu_pattern = False

        fpn_in_channels = in_channels
        fpn_strides = strides

        self.fnode = nn.ModuleList()
        reduction_base = strides[0]
        for i, fnode_cfg in fpn_config.items():
            if fnode_cfg['is_out'] == 1:
                fpn_channels = outer_fpn_channels
            else:
                fpn_channels = inner_fpn_channels

            reduction = fnode_cfg['reduction']
            fpn_channels_idx = int(math.log(reduction // reduction_base, 2))
            combine = GiraffeCombine(
                fpn_in_channels, fpn_strides, fpn_config, fpn_channels, tuple(fnode_cfg['inputs_offsets']),
                target_reduction=reduction, weight_method=fnode_cfg['weight_method'])

            after_combine = nn.Sequential()

            in_channel_sum = 0
            out_channels = 0
            for input_offset in fnode_cfg['inputs_offsets']:
                in_channel_sum += fpn_in_channels[input_offset]

            out_channels = fpn_channels[fpn_channels_idx]

            if merge_type == 'csp':
                after_combine.add_module('CspLayer', CSPLayer(in_channel_sum, out_channels, 2, shortcut=True, depthwise=False, act='silu'))
            elif merge_type == 'shuffle':
                after_combine.add_module('shuffleBlock', InvertedResidual(in_channel_sum, in_channel_sum, stride=1))
                after_combine.add_module('conv1x1', nn.Conv2d(in_channel_sum, out_channels, 1, 1, 0))
            elif merge_type == 'conv':
                after_combine.add_module('conv1x1', nn.Conv2d(in_channel_sum, out_channels, 1, 1, 0))
                after_combine.add_module('conv', DepthwiseConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='Swish')
                    if separable_conv else ConvModule(out_channels, out_channels, 3, 1, 1, norm_cfg=dict(type='BN'), activation='Swish'))

            self.fnode.append(GiraffeNode(combine=combine, after_combine=after_combine))
            # self.feature_info[i] = dict(num_chs=fpn_channels[fpn_channels_idx], reduction=reduction)
            fpn_in_channels.append(fpn_channels[fpn_channels_idx])
            fpn_strides.append(reduction)

    def forward(self, x):
        for fn in self.fnode:
            x.append(fn(x))
        return x[-self.num_levels::]


class GiraffeNeck(nn.Module):
    fpn_config = {3: {'reduction': 32, 'inputs_offsets': [2, 1], 'weight_method': 'concat', 'is_out': 0},
                 4: {'reduction': 16, 'inputs_offsets': [1, 3, 2, 0], 'weight_method': 'concat', 'is_out': 0},
                 5: {'reduction': 8, 'inputs_offsets': [0, 4, 1], 'weight_method': 'concat', 'is_out': 0},
                 6: {'reduction': 8, 'inputs_offsets': [5, 4], 'weight_method': 'concat', 'is_out': 0},
                 7: {'reduction': 16, 'inputs_offsets': [4, 6, 3, 5], 'weight_method': 'concat', 'is_out': 0},
                 8: {'reduction': 32, 'inputs_offsets': [3, 7, 4], 'weight_method': 'concat', 'is_out': 0},
                 9: {'reduction': 8, 'inputs_offsets': [6], 'weight_method': 'concat', 'is_out': 1},
                 10: {'reduction': 16, 'inputs_offsets': [7], 'weight_method': 'concat', 'is_out': 1},
                 11: {'reduction': 32, 'inputs_offsets': [8], 'weight_method': 'concat', 'is_out': 1}}
    def __init__(self, in_channels, fpn_channels=[96, 160, 384], out_channels=[96, 160, 384],
                 strides=[8, 16, 32], separable_conv=False,merge_type='csp', depth_mul=1.0, width_mul=1.0):
        super(GiraffeNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = list(map(lambda x: max(round(x * width_mul), 1), in_channels))
        self.out_channels = list(map(lambda x: max(round(x * width_mul), 1), out_channels))
        self.num_ins = len(in_channels)

        self.resample = nn.ModuleDict()
        self.cell = nn.Sequential()
        giraffe_layer = GiraffeLayer(
            in_channels=in_channels,
            strides=strides,
            fpn_config=self.fpn_config,
            inner_fpn_channels=fpn_channels,
            outer_fpn_channels=out_channels,
            separable_conv=separable_conv,
            merge_type=merge_type
        )
        self.cell.add_module('giraffeNeck', giraffe_layer)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03
            elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True


    def forward(self, x):
        assert len(x) == len(self.in_channels)
        for resample in self.resample.values():
            x.append(resample(x[-1]))
        out = self.cell(x)
        return out


if __name__ == "__main__":
    import torch
    in_channels = [128, 256, 512]
    out_channels = [96, 160, 384]
    scales = [80, 40, 20]
    inputs = [torch.rand(1, c, s, s) for c, s in zip(in_channels, scales)]
    for i in range(len(inputs)):
        print(f'inputs[{i}].shape = {inputs[i].shape}')
    self = GiraffeNeck(in_channels, out_channels, depth_mul=1, width_mul=1).eval()
    print(self)
    outputs = self.forward(inputs)
    for i in range(len(outputs)):
        print(f'outputs[{i}].shape = {outputs[i].shape}')