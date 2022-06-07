# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/19 15:37
# @Author : liumin
# @File : ppyoloe_modules.py
import copy

import torch
import torch.nn as nn
import torchvision
from torch import distributed as dist
import torch.nn.functional as F
import numpy as np


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def identity(x):
    return x


def mish(x):
    return F.mish(x) if hasattr(F, mish) else x * F.tanh(F.softplus(x))


def swish(x):
    return x * torch.sigmoid(x)

TRT_ACT_SPEC = {'swish': swish}

ACT_SPEC = {'mish': mish, 'swish': swish}


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return identity

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    if trt and name in TRT_ACT_SPEC:
        fn = TRT_ACT_SPEC[name]
    elif name in ACT_SPEC:
        fn = ACT_SPEC[name]
    else:
        fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.conv, torch.nn.Conv2d):
            if self.conv.weight.requires_grad:
                param_group_conv = {'params': [self.conv.weight]}
                param_group_conv['lr'] = base_lr * 1.0
                param_group_conv['base_lr'] = base_lr * 1.0
                param_group_conv['weight_decay'] = base_wd
                param_group_conv['need_clip'] = need_clip
                param_group_conv['clip_norm'] = clip_norm
                param_groups.append(param_group_conv)
        if self.bn is not None:
            if self.bn.weight.requires_grad:
                param_group_norm_weight = {'params': [self.bn.weight]}
                param_group_norm_weight['lr'] = base_lr * 1.0
                param_group_norm_weight['base_lr'] = base_lr * 1.0
                param_group_norm_weight['weight_decay'] = 0.0
                param_group_norm_weight['need_clip'] = need_clip
                param_group_norm_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_weight)
            if self.bn.bias.requires_grad:
                param_group_norm_bias = {'params': [self.bn.bias]}
                param_group_norm_bias['lr'] = base_lr * 1.0
                param_group_norm_bias['base_lr'] = base_lr * 1.0
                param_group_norm_bias['weight_decay'] = 0.0
                param_group_norm_bias['need_clip'] = need_clip
                param_group_norm_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_norm_bias)


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.act(x_se)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if isinstance(self.fc, torch.nn.Conv2d):
            if self.fc.weight.requires_grad:
                param_group_conv_weight = {'params': [self.fc.weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.fc.bias.requires_grad:
                param_group_conv_bias = {'params': [self.fc.bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if hasattr(self, 'conv'):
            self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        else:
            self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.copy_(kernel)
        self.conv.bias.copy_(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std



class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', shortcut=True):
        super(BasicBlock, self).__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



class CSPResStage(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca'):
        super(CSPResStage, self).__init__()

        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            BasicBlock(
                ch_mid // 2, ch_mid // 2, act=act, shortcut=True)
            for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.cat([y1, y2], 1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        if self.conv_down is not None:
            self.conv_down.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.blocks:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.attn is not None:
            self.attn.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class DropBlock(torch.nn.Module):
    def __init__(self, block_size, keep_prob, name, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890

        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = torch.rand(x.shape, device=x.device)
            matrix = (matrix < gamma).float()
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y


class SPP(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 k,
                 pool_size,
                 act='swish',
                 data_format='NCHW'):
        super(SPP, self).__init__()
        self.pool = []
        self.data_format = data_format
        for i, size in enumerate(pool_size):
            name = 'pool{}'.format(i)
            pool = nn.MaxPool2d(
                    kernel_size=size,
                    stride=1,
                    padding=size // 2,
                    ceil_mode=False)
            self.add_module(name, pool)
            self.pool.append(pool)
        self.conv = ConvBNLayer(ch_in, ch_out, k, padding=k // 2, act=act)

    def forward(self, x):
        outs = [x]
        for pool in self.pool:
            outs.append(pool(x))
        if self.data_format == 'NCHW':
            y = torch.cat(outs, 1)
        else:
            y = torch.cat(outs, -1)

        y = self.conv(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)



class CSPStage(nn.Module):
    def __init__(self, block_fn, ch_in, ch_out, n, act='swish', spp=False):
        super(CSPStage, self).__init__()

        ch_mid = int(ch_out // 2)
        self.conv1 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.conv2 = ConvBNLayer(ch_in, ch_mid, 1, act=act)
        self.convs = nn.Sequential()
        next_ch_in = ch_mid
        for i in range(n):
            self.convs.add_module(
                str(i),
                eval(block_fn)(next_ch_in, ch_mid, act=act, shortcut=False))
            if i == (n - 1) // 2 and spp:
                self.convs.add_module(
                    'spp', SPP(ch_mid * 4, ch_mid, 1, [5, 9, 13], act=act))
            next_ch_in = ch_mid
        self.conv3 = ConvBNLayer(ch_mid * 2, ch_out, 1, act=act)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.convs(y2)
        y = torch.cat([y1, y2], 1)
        y = self.conv3(y)
        return y

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv1.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv2.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for layer in self.convs:
            layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        self.conv3.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)


class CustomCSPPAN(nn.Module):
    __shared__ = ['norm_type', 'data_format', 'width_mult', 'depth_mult', 'trt']

    def __init__(self,
                 in_channels=[256, 512, 1024],
                 out_channels=[1024, 512, 256],
                 norm_type='bn',
                 act='leaky',
                 stage_fn='CSPStage',
                 block_fn='BasicBlock',
                 stage_num=1,
                 block_num=3,
                 drop_block=False,
                 block_size=3,
                 keep_prob=0.9,
                 spp=False,
                 data_format='NCHW',
                 width_mult=1.0,
                 depth_mult=1.0,
                 trt=False):

        super(CustomCSPPAN, self).__init__()
        out_channels = [max(round(c * width_mult), 1) for c in out_channels]
        block_num = max(round(block_num * depth_mult), 1)
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        self.num_blocks = len(in_channels)
        self.data_format = data_format
        self._out_channels = out_channels
        in_channels = in_channels[::-1]
        fpn_stages = []
        fpn_routes = []
        for i, (ch_in, ch_out) in enumerate(zip(in_channels, out_channels)):
            if i > 0:
                ch_in += ch_pre // 2

            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=(spp and i == 0)))

            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            fpn_stages.append(stage)

            if i < self.num_blocks - 1:
                fpn_routes.append(
                    ConvBNLayer(
                        ch_in=ch_out,
                        ch_out=ch_out // 2,
                        filter_size=1,
                        stride=1,
                        padding=0,
                        act=act))

            ch_pre = ch_out

        self.fpn_stages = nn.ModuleList(fpn_stages)
        self.fpn_routes = nn.ModuleList(fpn_routes)

        pan_stages = []
        pan_routes = []
        for i in reversed(range(self.num_blocks - 1)):
            pan_routes.append(
                ConvBNLayer(
                    ch_in=out_channels[i + 1],
                    ch_out=out_channels[i + 1],
                    filter_size=3,
                    stride=2,
                    padding=1,
                    act=act))

            ch_in = out_channels[i] + out_channels[i + 1]
            ch_out = out_channels[i]
            stage = nn.Sequential()
            for j in range(stage_num):
                stage.add_module(
                    str(j),
                    eval(stage_fn)(block_fn,
                                   ch_in if j == 0 else ch_out,
                                   ch_out,
                                   block_num,
                                   act=act,
                                   spp=False))
            if drop_block:
                stage.add_module('drop', DropBlock(block_size, keep_prob))

            pan_stages.append(stage)

        self.pan_stages = nn.ModuleList(pan_stages[::-1])
        self.pan_routes = nn.ModuleList(pan_routes[::-1])


    def forward(self, blocks, for_mot=False):
        blocks = blocks[::-1]
        fpn_feats = []

        for i, block in enumerate(blocks):
            if i > 0:
                block = torch.cat([route, block], 1)
            route = self.fpn_stages[i](block)
            fpn_feats.append(route)

            if i < self.num_blocks - 1:
                route = self.fpn_routes[i](route)
                route = F.interpolate(route, scale_factor=2.)

        pan_feats = [fpn_feats[-1], ]
        route = fpn_feats[-1]
        for i in reversed(range(self.num_blocks - 1)):
            block = fpn_feats[i]
            route = self.pan_routes[i](route)
            block = torch.cat([route, block], 1)
            route = self.pan_stages[i](block)
            pan_feats.append(route)

        return pan_feats[::-1]

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(self.num_blocks):
            for layer in self.fpn_stages[i]:
                layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if i < self.num_blocks - 1:
                self.fpn_routes[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        for i in reversed(range(self.num_blocks - 1)):
            self.pan_routes[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            for layer in self.pan_stages[i]:
                layer.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    # @property
    # def out_shape(self):
    #     return [ShapeSpec(channels=c) for c in self._out_channels]


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='swish'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        self.conv.add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
        if self.fc.weight.requires_grad:
            param_group_conv_weight = {'params': [self.fc.weight]}
            param_group_conv_weight['lr'] = base_lr * 1.0
            param_group_conv_weight['base_lr'] = base_lr * 1.0
            param_group_conv_weight['weight_decay'] = base_wd
            param_group_conv_weight['need_clip'] = need_clip
            param_group_conv_weight['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_weight)
        if self.fc.bias.requires_grad:
            param_group_conv_bias = {'params': [self.fc.bias]}
            param_group_conv_bias['lr'] = base_lr * 1.0
            param_group_conv_bias['base_lr'] = base_lr * 1.0
            param_group_conv_bias['weight_decay'] = base_wd
            param_group_conv_bias['need_clip'] = need_clip
            param_group_conv_bias['clip_norm'] = clip_norm
            param_groups.append(param_group_conv_bias)


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = F.relu(xkis2 - xkis1)
        h_inter = F.relu(ykis2 - ykis1)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        # x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        # x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        anchor = torch.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            -1).to(feat.dtype)
        anchor_point = torch.stack(
            [shift_x, shift_y], -1).to(feat.dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            torch.full(
                [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
    anchors = torch.cat(anchors)
    anchors.requires_grad_(False)
    anchor_points = torch.cat(anchor_points)
    anchor_points.requires_grad_(False)
    stride_tensor = torch.cat(stride_tensor)
    stride_tensor.requires_grad_(False)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox



class PPYOLOEHead(nn.Module):
    __shared__ = ['num_classes', 'eval_size', 'trt', 'exclude_nms']
    __inject__ = ['static_assigner', 'assigner', 'nms']

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='swish',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner='ATSSAssigner',
                 assigner='TaskAlignedAssigner',
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 trt=False,
                 nms_cfg=None,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        static_assigner_params = {'topk': 9, 'num_classes': 80}
        assigner_params =  {'topk': 13, 'alpha': 1.0, 'beta': 6.0}
        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = ATSSAssigner(**static_assigner_params)
        self.assigner = TaskAlignedAssigner(**assigner_params)
        self.nms = nms
        # if isinstance(self.nms, MultiClassNMS) and trt:
        #     self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.nms_cfg = nms_cfg
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()

    def add_param_group(self, param_groups, base_lr, base_wd, need_clip, clip_norm):
        for i in range(len(self.in_channels)):
            self.stem_cls[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            self.stem_reg[i].add_param_group(param_groups, base_lr, base_wd, need_clip, clip_norm)
            if self.pred_cls[i].weight.requires_grad:
                param_group_conv_weight = {'params': [self.pred_cls[i].weight]}
                param_group_conv_weight['lr'] = base_lr * 1.0
                param_group_conv_weight['base_lr'] = base_lr * 1.0
                param_group_conv_weight['weight_decay'] = base_wd
                param_group_conv_weight['need_clip'] = need_clip
                param_group_conv_weight['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight)
            if self.pred_cls[i].bias.requires_grad:
                param_group_conv_bias = {'params': [self.pred_cls[i].bias]}
                param_group_conv_bias['lr'] = base_lr * 1.0
                param_group_conv_bias['base_lr'] = base_lr * 1.0
                param_group_conv_bias['weight_decay'] = base_wd
                param_group_conv_bias['need_clip'] = need_clip
                param_group_conv_bias['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias)
            if self.pred_reg[i].weight.requires_grad:
                param_group_conv_weight2 = {'params': [self.pred_reg[i].weight]}
                param_group_conv_weight2['lr'] = base_lr * 1.0
                param_group_conv_weight2['base_lr'] = base_lr * 1.0
                param_group_conv_weight2['weight_decay'] = base_wd
                param_group_conv_weight2['need_clip'] = need_clip
                param_group_conv_weight2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_weight2)
            if self.pred_reg[i].bias.requires_grad:
                param_group_conv_bias2 = {'params': [self.pred_reg[i].bias]}
                param_group_conv_bias2['lr'] = base_lr * 1.0
                param_group_conv_bias2['base_lr'] = base_lr * 1.0
                param_group_conv_bias2['weight_decay'] = base_wd
                param_group_conv_bias2['need_clip'] = need_clip
                param_group_conv_bias2['clip_norm'] = clip_norm
                param_groups.append(param_group_conv_bias2)

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {'in_channels': [i.channels for i in input_shape], }

    def _init_weights(self):
        bias_cls = float(-np.log((1 - 0.01) / 0.01))
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            nn.init.constant_(cls_.weight, 0)
            nn.init.constant_(cls_.bias, bias_cls)
            nn.init.constant_(reg_.weight, 0)
            nn.init.constant_(reg_.bias, 1.0)

        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def forward_train(self, feats, targets):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        cls_score_list = torch.cat(cls_score_list, 1)
        reg_distri_list = torch.cat(reg_distri_list, 1)

        losses = self.get_loss([
            cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor
        ], targets)
        return losses

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], -1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    [h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward_eval(self, feats):
        if self.eval_size:
            anchor_points, stride_tensor = self.anchor_points, self.stride_tensor
        else:
            anchor_points, stride_tensor = self._generate_anchors(feats)
        cls_score_list, reg_dist_list = [], []
        for i, feat in enumerate(feats):
            b, _, h, w = feat.shape
            l = h * w
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_dist = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            reg_dist = reg_dist.reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.reshape([b, self.num_classes, l]))
            reg_dist_list.append(reg_dist.reshape([b, 4, l]))

        cls_score_list = torch.cat(cls_score_list, -1)  # [N, 80, A]
        reg_dist_list = torch.cat(reg_dist_list, -1)    # [N,  4, A]

        return cls_score_list, reg_dist_list, anchor_points, stride_tensor

    def forward(self, feats, targets=None):
        assert len(feats) == len(self.fpn_strides), \
            "The size of feats is not equal to size of fpn_strides"

        if self.training:
            return self.forward_train(feats, targets)
        else:
            return self.forward_eval(feats)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        # loss = F.binary_cross_entropy(
        #     score, label, weight=weight, reduction='sum')

        score = score.to(torch.float32)
        eps = 1e-9
        loss = label * (0 - torch.log(score + eps)) + \
               (1.0 - label) * (0 - torch.log(1.0 - score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # loss = F.binary_cross_entropy(
        #     pred_score, gt_score, weight=weight, reduction='sum')

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = pred_score.to(torch.float32)
        eps = 1e-9
        loss = gt_score * (0 - torch.log(pred_score + eps)) + \
               (1.0 - gt_score) * (0 - torch.log(1.0 - pred_score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = pred_dist.shape
        device = pred_dist.device
        pred_dist = pred_dist.reshape([b, l, 4, self.reg_max + 1])
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.matmul(self.proj.to(device))
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = F.softmax(pred_dist, dim=-1)
        target_left_onehot = F.one_hot(target_left, pred_dist_act.shape[-1])
        target_right_onehot = F.one_hot(target_right, pred_dist_act.shape[-1])
        loss_left = target_left_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_left = loss_left.sum(-1) * weight_left
        loss_right = loss_right.sum(-1) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([]).to(pred_dist.device)
            loss_iou = torch.zeros([]).to(pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
            # loss_l1 = None
            # loss_iou = None
            # loss_dfl = None
        return loss_l1, loss_iou, loss_dfl

    def get_loss(self, head_outs, gt_meta):
        pred_scores, pred_distri, anchors,\
        anchor_points, num_anchors_list, stride_tensor = head_outs
        device = pred_scores.device
        anchors = anchors.to(device)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_labels = gt_meta['labels']
        gt_labels = gt_labels.to(torch.int64)
        gt_bboxes = gt_meta['boxes']
        gt_labels = gt_labels.unsqueeze(2)
        pad_gt_mask = torch.zeros_like(gt_labels)
        pad_gt_mask[gt_labels >= 0] = 1

        gt_meta['epoch_id'] = 1

        # miemie2013: 剪掉填充的gt
        num_boxes = pad_gt_mask.sum([1, 2])
        num_max_boxes = num_boxes.max().to(torch.int32)
        pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
        gt_labels = gt_labels[:, :num_max_boxes, :]
        gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

        # label assignment
        if gt_meta['epoch_id'] < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25
        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        # 每张卡上的assigned_scores_sum求平均，而且max(x, 1)
        assigned_scores_sum = assigned_scores.sum()
        world_size = get_world_size()
        if world_size > 1:
            dist.all_reduce(assigned_scores_sum, op=dist.ReduceOp.SUM)
            assigned_scores_sum = assigned_scores_sum / world_size
        assigned_scores_sum = F.relu(assigned_scores_sum - 1.) + 1.  # y = max(x, 1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl
        out_dict = {
            'loss': loss,
            'loss_cls': loss_cls,
            'loss_iou': loss_iou,
            'loss_dfl': loss_dfl,
            'loss_l1': loss_l1,
        }
        return out_dict

    def post_process_org(self, head_outs, scale_factor):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist.permute((0, 2, 1)))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        scale_y, scale_x = torch.split(scale_factor, 1, -1)
        scale_factor = torch.cat(
            [scale_x, scale_y, scale_x, scale_y], -1).reshape([-1, 1, 4])
        pred_bboxes /= scale_factor   # [N, A, 4]     pred_scores.shape = [N, 80, A]
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            # nms
            preds = []
            nms_cfg = copy.deepcopy(self.nms_cfg)
            nms_type = nms_cfg.pop('nms_type')
            batch_size = pred_bboxes.shape[0]
            yolo_scores = pred_scores.permute((0, 2, 1))  #  [N, A, 80]
            if nms_type == 'matrix_nms':
                for i in range(batch_size):
                    pred = matrix_nms(pred_bboxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                    preds.append(pred)
            elif nms_type == 'multiclass_nms':
                preds = my_multiclass_nms(pred_bboxes, yolo_scores, **nms_cfg)
            return preds
            # bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            # return bbox_pred, bbox_num

    def post_process(self, head_outs):
        pred_scores, pred_dist, anchor_points, stride_tensor = head_outs
        pred_bboxes = batch_distance2bbox(anchor_points, pred_dist.permute((0, 2, 1)))
        pred_bboxes *= stride_tensor
        # scale bbox to origin
        '''
            # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
            scale_y, scale_x = torch.split(scale_factor, 1, -1)
            scale_factor = torch.cat([scale_x, scale_y, scale_x, scale_y], -1).reshape([-1, 1, 4])
            pred_bboxes /= scale_factor   # [N, A, 4]     pred_scores.shape = [N, 80, A]
        '''
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_bboxes.sum(), pred_scores.sum()
        else:
            # nms
            preds = []
            nms_cfg = copy.deepcopy(self.nms_cfg)
            nms_type = nms_cfg.pop('nms_type')
            batch_size = pred_bboxes.shape[0]
            yolo_scores = pred_scores.permute((0, 2, 1))  #  [N, A, 80]
            if nms_type == 'matrix_nms':
                for i in range(batch_size):
                    pred = matrix_nms(pred_bboxes[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                    preds.append(pred)
            elif nms_type == 'multiclass_nms':
                preds = my_multiclass_nms(pred_bboxes, yolo_scores, **nms_cfg)
            return preds
            # bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            # return bbox_pred, bbox_num


def my_multiclass_nms(bboxes, scores, score_threshold=0.7, nms_threshold=0.45, nms_top_k=1000, keep_top_k=100, class_agnostic=False):
    '''
    :param bboxes:   shape = [N, A,  4]   "左上角xy + 右下角xy"格式
    :param scores:   shape = [N, A, 80]
    :param score_threshold:
    :param nms_threshold:
    :param nms_top_k:
    :param keep_top_k:
    :param class_agnostic:
    :return:
    '''

    # 每张图片的预测结果
    output = [None for _ in range(len(bboxes))]
    # 每张图片分开遍历
    for i, (xyxy, score) in enumerate(zip(bboxes, scores)):
        '''
        :var xyxy:    shape = [A, 4]   "左上角xy + 右下角xy"格式
        :var score:   shape = [A, 80]
        '''

        # 每个预测框最高得分的分数和对应的类别id
        class_conf, class_pred = torch.max(score, 1, keepdim=True)

        # 分数超过阈值的预测框为True
        conf_mask = (class_conf.squeeze() >= score_threshold).squeeze()
        # 这样排序 (x1, y1, x2, y2, 得分, 类别id)
        detections = torch.cat((xyxy, class_conf, class_pred.float()), 1)
        # 只保留超过阈值的预测框
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # 使用torchvision自带的nms、batched_nms
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_threshold,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_threshold,
            )

        detections = detections[nms_out_index]

        # 保留得分最高的keep_top_k个
        sort_inds = torch.argsort(detections[:, 4], descending=True)
        if keep_top_k > 0 and len(sort_inds) > keep_top_k:
            sort_inds = sort_inds[:keep_top_k]
        detections = detections[sort_inds, :]

        # 为了保持和matrix_nms()一样的返回风格 cls、score、xyxy。
        detections = torch.cat((detections[:, 5:6], detections[:, 4:5], detections[:, :4]), 1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]



def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    iou_matrix = iou_matrix.triu(diagonal=1)   # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)   # shape: [n_samples, ]
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update





def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    inds = (scores > score_threshold)
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0

    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    bboxes = bboxes[inds[:, 0]]

    # sort and keep top nms_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    bboxes = bboxes[keep, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep keep_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = cate_scores.unsqueeze(1)
    cate_labels = cate_labels.unsqueeze(1).float()
    pred = torch.cat([cate_labels, cate_scores, bboxes], 1)

    return pred


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)

def index_sample_2d(tensor, index):
    assert tensor.ndim == 2
    assert index.ndim == 2
    assert index.dtype == torch.int64
    d0, d1 = tensor.shape
    d2, d3 = index.shape
    assert d0 == d2
    tensor_ = tensor.reshape((-1, ))
    batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    batch_ind = batch_ind.to(index.device)
    index_ = index + batch_ind
    index_ = index_.reshape((-1, ))
    out = tensor_[index_]
    out = out.reshape((d2, d3))
    return out


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool)
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, -1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = torch.topk(
                distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask, topk_idxs,
                                     torch.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
            is_in_topk = torch.where(is_in_topk > 1,
                                      torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.cat(is_in_topk_list, -1)
        topk_idxs_list = torch.cat(topk_idxs_list, -1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        batch_anchor_bboxes = anchor_bboxes.unsqueeze(0).repeat([batch_size, 1, 1])
        ious = iou_similarity(gt_bboxes, batch_anchor_bboxes)

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        aaaaaa1 = iou_candidates.reshape((-1, iou_candidates.shape[-1]))
        aaaaaa2 = topk_idxs.reshape((-1, topk_idxs.shape[-1]))
        iou_threshold = index_sample_2d(aaaaaa1, aaaaaa2)
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + \
                        iou_threshold.std(dim=-1, keepdim=True)
        is_in_topk = torch.where(
            iou_candidates > iou_threshold.repeat([1, 1, num_anchors]),
            is_in_topk, torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            # when use fp16
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive.to(is_max_iou.dtype))
            # mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).repeat(
                [1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(assigned_gt_index.device)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gather_1d(
            gt_labels.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gather_1d(
            gt_bboxes.reshape([-1, 4]), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        assigned_scores = assigned_scores.to(torch.float32)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, dim=-1, index=torch.Tensor(ind).to(torch.int32).to(assigned_scores.device))
        if pred_bboxes is not None:
            # assigned iou
            ious = iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious_max, _ = ious.max(-2)
            ious_max = ious_max.unsqueeze(-1)
            assigned_scores *= ious_max
        elif gt_scores is not None:
            gather_scores = gather_1d(
                gt_scores.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores,
                                         torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)
        # if torch.isnan(assigned_scores).any():
        #     print()
        return assigned_labels, assigned_bboxes, assigned_scores


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.permute([0, 2, 1])
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(gt_labels.device)
        gt_labels_ind = torch.stack([batch_ind.repeat([1, num_max_boxes]), gt_labels.squeeze(-1)], -1)
        bbox_cls_scores = gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool))

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gather_1d(gt_labels.flatten(), assigned_gt_index.flatten())
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gather_1d(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten())
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, dim=-1, index=torch.Tensor(ind).to(torch.int32).to(assigned_scores.device))
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = alignment_metrics.max(-1, keepdim=True)
        max_ious_per_instance, _ = (ious * mask_positive).max(-1, keepdim=True)
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics, _ = alignment_metrics.max(-2)
        alignment_metrics = alignment_metrics.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores


def gather_1d(tensor, index):
    assert index.ndim == 1
    assert index.dtype == torch.int64
    # d0, d1 = tensor.shape
    # d2, d3 = index.shape
    # assert d0 == d2
    # tensor_ = tensor.reshape((-1, ))
    # batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    # index_ = index + batch_ind
    # index_ = index_.reshape((-1, ))
    # out = tensor_[index_]
    out = tensor[index]
    return out



def gather_nd(tensor, index):
    if tensor.ndim == 4 and index.ndim == 2:
        N, R, S, T = tensor.shape
        index_0 = index[:, 0]  # [M, ]
        index_1 = index[:, 1]  # [M, ]
        index_2 = index[:, 2]  # [M, ]
        index_ = index_0 * R * S + index_1 * S + index_2  # [M, ]
        x2 = torch.reshape(tensor, (N * R * S, T))  # [N*R*S, T]
        index_ = index_.to(torch.int64)
        out = gather_1d(x2, index_)
    elif tensor.ndim == 3 and index.ndim == 3:
        A, B, C = tensor.shape
        D, E, F = index.shape
        assert F == 2
        # out.shape = [D, E, C]
        tensor_ = tensor.reshape((-1, C))   # [A*B, C]
        index_ = index.reshape((-1, F))     # [D*E, F]


        index_0 = index_[:, 0]  # [D*E, ]
        index_1 = index_[:, 1]  # [D*E, ]
        index_ = index_0 * B + index_1  # [D*E, ]

        out = gather_1d(tensor_, index_)  # [D*E, C]
        out = out.reshape((D, E, C))   # [D, E, C]
    else:
        raise NotImplementedError("not implemented.")
    return out


def pad_gt(gt_labels, gt_bboxes, gt_scores=None):
    r""" Pad 0 in gt_labels and gt_bboxes.
    Args:
        gt_labels (Tensor|List[Tensor], int64): Label of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 1], [n_2, 1], ...], here n = sum(n_i)
        gt_bboxes (Tensor|List[Tensor], float32): Ground truth bboxes,
            shape is [B, n, 4] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
        gt_scores (Tensor|List[Tensor]|None, float32): Score of gt_bboxes,
            shape is [B, n, 1] or [[n_1, 4], [n_2, 4], ...], here n = sum(n_i)
    Returns:
        pad_gt_labels (Tensor, int64): shape[B, n, 1]
        pad_gt_bboxes (Tensor, float32): shape[B, n, 4]
        pad_gt_scores (Tensor, float32): shape[B, n, 1]
        pad_gt_mask (Tensor, float32): shape[B, n, 1], 1 means bbox, 0 means no bbox
    """
    if isinstance(gt_labels, torch.Tensor) and isinstance(gt_bboxes, torch.Tensor):
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3
        pad_gt_mask = (
            gt_bboxes.sum(axis=-1, keepdim=True) > 0).to(gt_bboxes.dtype)
        if gt_scores is None:
            gt_scores = pad_gt_mask.clone()
        assert gt_labels.ndim == gt_scores.ndim

        return gt_labels, gt_bboxes, gt_scores, pad_gt_mask
    elif isinstance(gt_labels, list) and isinstance(gt_bboxes, list):
        assert len(gt_labels) == len(gt_bboxes), \
            'The number of `gt_labels` and `gt_bboxes` is not equal. '
        num_max_boxes = max([len(a) for a in gt_bboxes])
        batch_size = len(gt_bboxes)
        # pad label and bbox
        pad_gt_labels = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_labels[0].dtype)
        pad_gt_bboxes = paddle.zeros(
            [batch_size, num_max_boxes, 4], dtype=gt_bboxes[0].dtype)
        pad_gt_scores = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        pad_gt_mask = paddle.zeros(
            [batch_size, num_max_boxes, 1], dtype=gt_bboxes[0].dtype)
        for i, (label, bbox) in enumerate(zip(gt_labels, gt_bboxes)):
            if len(label) > 0 and len(bbox) > 0:
                pad_gt_labels[i, :len(label)] = label
                pad_gt_bboxes[i, :len(bbox)] = bbox
                pad_gt_mask[i, :len(bbox)] = 1.
                if gt_scores is not None:
                    pad_gt_scores[i, :len(gt_scores[i])] = gt_scores[i]
        if gt_scores is None:
            pad_gt_scores = pad_gt_mask.clone()
        return pad_gt_labels, pad_gt_bboxes, pad_gt_scores, pad_gt_mask
    else:
        raise ValueError('The input `gt_labels` or `gt_bboxes` is invalid! ')


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(
        metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1,
                              torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


def check_points_inside_bboxes(points,
                               bboxes,
                               center_radius_tensor=None,
                               eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze(0).unsqueeze(1)
    x, y = points.chunk(2, axis=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = torch.cat([l, t, r, b], -1)
    delta_ltrb_min, _ = delta_ltrb.min(-1)
    is_in_bboxes = (delta_ltrb_min > eps)
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze(0).unsqueeze(1)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.cat([l, t, r, b], -1)
        delta_ltrb_c_min = delta_ltrb_c.min(-1)
        is_in_center = (delta_ltrb_c_min > eps)
        return (torch.logical_and(is_in_bboxes, is_in_center),
                torch.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.to(bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute((0, 2, 1))
    return is_max_iou.to(ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(axis=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.to(ious.dtype)



def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    bboxes_a = bboxes_a.to(torch.float32)
    bboxes_b = bboxes_b.to(torch.float32)
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = torch.cat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                           bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], dim=-1)
        box_b = torch.cat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                           bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], dim=-1)

    box_a_rb = torch.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    box_a_rb = torch.tile(box_a_rb, [1, 1, B, 1])
    box_b_rb = torch.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = torch.tile(box_b_rb, [1, A, 1, 1])
    max_xy = torch.minimum(box_a_rb, box_b_rb)

    box_a_lu = torch.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = torch.tile(box_a_lu, [1, 1, B, 1])
    box_b_lu = torch.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = torch.tile(box_b_lu, [1, A, 1, 1])
    min_xy = torch.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = torch.reshape(area_a, (N, A, 1))
    area_a = torch.tile(area_a, [1, 1, B])  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = torch.reshape(area_b, (N, 1, B))
    area_b = torch.tile(area_b, [1, A, 1])  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]

def iou_similarity(box1, box2):
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    box1 = box1.to(torch.float32)
    box2 = box2.to(torch.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)