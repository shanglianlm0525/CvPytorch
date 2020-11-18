# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/18 13:10
# @Author : liumin
# @File : fuse.py

import torch
from torch import nn

torch.set_grad_enabled(False)

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)

    def forward(self, x):
        return self.bn(self.conv(x))

    def fuseforward(self, x):
        return self.conv(x)


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def fuse(model):  # fuse model Conv2d() + BatchNorm2d() layers
    # print('Fusing layers... ')
    for m in model.modules():
        if type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            m.bn = None  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    return model


if __name__=='__main__':
    x = torch.randn(2, 64, 256, 256)

    net = nn.Sequential(
        Conv(64, 32, 1, 1),
        Conv(32, 128, 3, 1, g=1)
    )
    net.eval()

    y1 = net.forward(x)
    fused_net = fuse(net)
    y2 = fused_net.forward(x)
    d = (y1 - y2).norm().div(y1.norm()).item()
    print("error: %.8f" % d)