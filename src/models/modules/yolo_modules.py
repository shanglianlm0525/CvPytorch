# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/21 9:18
# @Author : liumin
# @File : yolo_modules.py
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from src.base.base_module import BaseModule
from src.models.bricks import ConvModule, DepthwiseSeparableConvModule



class Focus(BaseModule):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, kernel_sizes=1, stride=1,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='Swish')):
        super().__init__()
        self.conv = ConvModule(in_channels * 4, out_channels, kernel_sizes, stride, padding=(kernel_sizes - 1) // 2,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)


class DarknetBottleneck(BaseModule):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (int): The kernel size of the convolution. Default: 0.5
        add_identity (bool): Whether to add identity to the out.
            Default: True
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Default: False
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish').
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=0.5,
                 shortcut=True,
                 depthwise=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Swish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        hidden_channels = int(out_channels * expansion)
        conv = DepthwiseSeparableConvModule if depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.shortcut = \
            shortcut and in_channels == out_channels

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.shortcut:
            return out + identity
        else:
            return out


class CSPLayer(BaseModule):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self, in_channels, out_channels, n=1, expansion=0.5, shortcut=True, depthwise=False,
        conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='Swish')
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(in_channels, hidden_channels, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = ConvModule(2 * hidden_channels, out_channels, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.m = nn.Sequential(*[
            DarknetBottleneck(hidden_channels, hidden_channels, 1.0, shortcut, depthwise,
                              conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
            for _ in range(n)
        ])

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class UpsamplingModule(BaseModule):
    def __init__(self, c1, c2, layer=3, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU')):
        super(UpsamplingModule, self).__init__()
        self.conv = ConvModule(c1, c2, 1, 1, 0, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.fuse = CSPLayer(c2 * 2, c2, n=layer, shortcut=False, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, y):
        x_conv = self.conv(x)
        return self.fuse(torch.cat([self.up(x_conv), y], dim=1)), x_conv


class DownsamplingModule(nn.Module):
    def __init__(self, c1, c2, layer=3, conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU')):
        super(DownsamplingModule, self).__init__()
        self.down = ConvModule(c1, c1, 3, 2, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.fuse = CSPLayer(c1 * 2, c2, n=layer, shortcut=False, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, y):
        return self.fuse(torch.cat([self.down(x), y], dim=1))


class SPPF(BaseModule):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13),
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='Swish'), init_cfg=None):
        super().__init__(init_cfg)
        self.kernel_sizes = kernel_sizes
        hidden_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, hidden_channels, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        if isinstance(kernel_sizes, int):
            self.m = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
        else:
            self.m = nn.ModuleList(
                [nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes]
            )
        self.conv2 = ConvModule(hidden_channels * 4, out_channels, 1, stride=1,
                                conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.m(x)
            y2 = self.m(y1)
            x = torch.cat([x, y1, y2, self.m(y2)], dim=1)
        else:
            x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class SimSPPF(BaseModule):
    '''Simplified SPPF with ReLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=5,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='Swish'), init_cfg=None):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(c_ * 4, out_channels, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))




class SimCSPSPPF(BaseModule):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='Swish'), init_cfg=None):
        super(SimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(in_channels, c_, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(c_, c_, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv4 = ConvModule(c_, c_, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = ConvModule(4 * c_, c_, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv6 = ConvModule(c_, c_, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv7 = ConvModule(2 * c_, out_channels, 1, stride=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class BiFusion(nn.Module):
    '''BiFusion Block in PAN'''

    def __init__(self, in_channels, out_channels,
                 conv_cfg=None, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='ReLU'), init_cfg=None):
        super().__init__()
        self.cv1 = ConvModule(in_channels[0], out_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(in_channels[1], out_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(out_channels * 3, out_channels, 1, 1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.upsample = nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels,
            kernel_size = 2, stride = 2, bias = True)
        self.downsample = ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1,
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(torch.cat((x0, x1, x2), dim=1))

#### repVGG

class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_cfg = dict(type='BN', requires_grad=True)

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(in_channels, out_channels, kernel_size, stride, padding, groups=groups, norm_cfg=norm_cfg, act_cfg=None)
            self.rbr_1x1 = ConvModule(in_channels, out_channels, 1, stride, padding_11, groups=groups, norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
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

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels, out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation, groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ConvWrapper(nn.Module):
    '''Normal Conv with SiLU activation'''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Conv_C3(nn.Module):
    '''Standard convolution in BepC3-Block'''
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k//2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class BepC3(nn.Module):
    '''Beer-mug RepC3 Block'''
    def __init__(self, in_channels, out_channels, n=1, e=0.5, concat=True, block=RepVGGBlock):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())

        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))


class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    def __init__(self, in_channels, out_channels, n=1, e=None, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()

        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):

    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs


class BottleRepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''

    def __init__(self, in_channels, out_channels, n=1):
        super().__init__()
        self.conv1 = BottleRep(in_channels, out_channels, weight=True)
        n = n // 2
        self.block = nn.Sequential(*(BottleRep(out_channels, out_channels, weight=True) for _ in range(n - 1))) if n > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class CSPStackRep(nn.Module):
    '''Beer-mug RepC3 Block'''

    def __init__(self, in_channels, out_channels, n=1, e=0.5, concat=True,
                 norm_cfg = dict(type='BN', requires_grad=True), act_cfg = dict(type='SiLU', inplace=True)):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvModule(in_channels, c_, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv2 = ConvModule(in_channels, c_, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.cv3 = ConvModule(2 * c_, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.m = BottleRepBlock(in_channels=c_, out_channels=c_, n=n)
        self.concat = concat
        if not concat:
            self.cv3 = ConvModule(c_, out_channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))



class DownA(nn.Module):
    def __init__(self, c1, c2, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True)):
        super(DownA, self).__init__()
        self.branch1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.branch2 = nn.Sequential(ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                     ConvModule(c2, c2, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat([x1, x2], dim=1)


class DownB(nn.Module):
    def __init__(self, c1, c2, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True)):
        super(DownB, self).__init__()
        self.branch1 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.branch2 = nn.Sequential(ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                     ConvModule(c2, c2, 3, 2, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))

    def forward(self, x, y):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        return torch.cat([x1, x2, y], dim=1)


class EELAN(nn.Module):
    def __init__(self, c1, c2, c3, norm_cfg=dict(type='BN', requires_grad=True), act_cfg=dict(type='SiLU', inplace=True)):
        super(EELAN, self).__init__()
        self.conv1 = ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv2 = ConvModule(c1, c2, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv3 = nn.Sequential(ConvModule(c2, c2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                   ConvModule(c2, c2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.conv4 = nn.Sequential(ConvModule(c2, c2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg),
                                   ConvModule(c2, c2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))

        self.conv5 = ConvModule(c2*4, c3, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        cat_x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv5(cat_x)