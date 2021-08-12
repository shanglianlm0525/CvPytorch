# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/10 20:09
# @Author : liumin
# @File : mlps.py

''' copied from https://github.com/xmu-xiaoma666/External-Attention-pytorch/tree/master/mlp '''

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


__all__ = ['ResMLP', 'gMLP', 'MlpMixer', 'RepMLP']

class Rearange(nn.Module):
    def __init__(self, image_size=14, patch_size=7):
        self.h = patch_size
        self.w = patch_size
        self.nw = image_size // patch_size
        self.nh = image_size // patch_size

        num_patches = (image_size // patch_size) ** 2
        super().__init__()

    def forward(self, x):
        ### bs,c,H,W
        bs, c, H, W = x.shape

        y = x.reshape(bs, c, self.h, self.nh, self.w, self.nw)
        y = y.permute(0, 3, 5, 2, 4, 1)  # bs,nh,nw,h,w,c
        y = y.contiguous().view(bs, self.nh * self.nw, -1)  # bs,nh*nw,h*w*c
        return y

class Affine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, channel))
        self.b = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        return x * self.g + self.b


class PreAffinePostLayerScale(nn.Module):  # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Module):
    '''
        ResMLP: Feedforward networks for image classification with data-efficient training
        https://arxiv.org/pdf/2105.03404.pdf
    '''
    def __init__(self, dim=128, image_size=14, patch_size=7, expansion_factor=4, depth=4, class_num=1000):
        super().__init__()
        self.flatten = Rearange(image_size, patch_size)
        num_patches = (image_size // patch_size) ** 2
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)
        self.embedding = nn.Linear((patch_size ** 2) * 3, dim)
        self.mlp = nn.Sequential()

        for i in range(depth):
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Conv1d(patch_size ** 2, patch_size ** 2, 1)))
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Linear(dim * expansion_factor, dim)
            )))

        self.aff = Affine(dim)

        self.classifier = nn.Linear(dim, class_num)
        self.softmax = nn.Softmax(1)

    def forward(self, x):
        y = self.flatten(x)
        y = self.embedding(y)
        y = self.mlp(y)
        y = self.aff(y)
        y = torch.mean(y, dim=1)  # bs,dim
        out = self.softmax(self.classifier(y))
        return out



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, len_sen):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(len_sen, len_sen, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        res, gate = torch.chunk(x, 2, -1)  # bs,n,d_ff
        ###Norm
        gate = self.ln(gate)  # bs,n,d_ff
        ###Spatial Proj
        gate = self.proj(gate)  # bs,n,d_ff
        return res * gate


class gMLP(nn.Module):
    '''
        Pay Attention to MLPs
        https://arxiv.org/abs/2105.08050
    '''
    def __init__(self, num_tokens=None, len_sen=49, dim=512, d_ff=1024, num_layers=6):
        super().__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_tokens, dim) if num_tokens is not None else nn.Identity()

        self.gmlp = nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d' % i, nn.LayerNorm(dim)),
            ('fc1_%d' % i, nn.Linear(dim, d_ff * 2)),
            ('gelu_%d' % i, nn.GELU()),
            ('sgu_%d' % i, SpatialGatingUnit(d_ff, len_sen)),
            ('fc2_%d' % i, nn.Linear(d_ff, dim)),
        ]))) for i in range(num_layers)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens),
            nn.Softmax(-1)
        )

    def forward(self, x):
        # embedding
        embeded = self.embedding(x)
        # gMLP
        y = nn.Sequential(*self.gmlp)(embeded)
        # to logits
        logits = self.to_logits(y)
        return logits


class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # x: (bs,tokens,channels) or (bs,channels,tokens)
        return self.fc2(self.gelu(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, tokens_mlp_dim=16, channels_mlp_dim=1024, tokens_hidden_dim=32, channels_hidden_dim=1024):
        super().__init__()
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp_block = MlpBlock(tokens_mlp_dim, mlp_dim=tokens_hidden_dim)
        self.channels_mlp_block = MlpBlock(channels_mlp_dim, mlp_dim=channels_hidden_dim)

    def forward(self, x):
        """
        x: (bs,tokens,channels)
        """
        ### tokens mixing
        y = self.ln(x)
        y = y.transpose(1, 2)  # (bs,channels,tokens)
        y = self.tokens_mlp_block(y)  # (bs,channels,tokens)
        ### channels mixing
        y = y.transpose(1, 2)  # (bs,tokens,channels)
        out = x + y  # (bs,tokens,channels)
        y = self.ln(out)  # (bs,tokens,channels)
        y = out + self.channels_mlp_block(y)  # (bs,tokens,channels)
        return y


class MlpMixer(nn.Module):
    '''
        MLP-Mixer: An all-MLP Architecture for Vision
        https://arxiv.org/pdf/2105.01601.pdf
    '''
    def __init__(self, num_classes, num_blocks, patch_size, tokens_hidden_dim, channels_hidden_dim, tokens_mlp_dim,
                 channels_mlp_dim):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks  # num of mlp layers
        self.patch_size = patch_size
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim
        self.embd = nn.Conv2d(3, channels_mlp_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(channels_mlp_dim)
        self.mlp_blocks = []
        for _ in range(num_blocks):
            self.mlp_blocks.append(MixerBlock(tokens_mlp_dim, channels_mlp_dim, tokens_hidden_dim, channels_hidden_dim))
        self.fc = nn.Linear(channels_mlp_dim, num_classes)

    def forward(self, x):
        y = self.embd(x)  # bs,channels,h,w
        bs, c, h, w = y.shape
        y = y.view(bs, c, -1).transpose(1, 2)  # bs,tokens,channels

        if (self.tokens_mlp_dim != y.shape[1]):
            raise ValueError('Tokens_mlp_dim is not correct.')

        for i in range(self.num_blocks):
            y = self.mlp_blocks[i](y)  # bs,tokens,channels
        y = self.ln(y)  # bs,tokens,channels
        y = torch.mean(y, dim=1, keepdim=False)  # bs,channels
        probs = self.fc(y)  # bs,num_classes
        return probs


class RepMLP(nn.Module):
    '''
        RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
        https://arxiv.org/pdf/2105.01883v1.pdf
    '''
    def __init__(self, C, O, H, W, h, w, fc1_fc2_reduction=1, fc3_groups=8, repconv_kernels=None, deploy=False):
        super().__init__()
        self.C = C
        self.O = O
        self.H = H
        self.W = W
        self.h = h
        self.w = w
        self.fc1_fc2_reduction = fc1_fc2_reduction
        self.repconv_kernels = repconv_kernels
        self.h_part = H // h
        self.w_part = W // w
        self.deploy = deploy
        self.fc3_groups = fc3_groups

        # make sure H,W can divided by h,w respectively
        assert H % h == 0
        assert W % w == 0

        self.is_global_perceptron = (H != h) or (W != w)
        ### global perceptron
        if (self.is_global_perceptron):
            if (not self.deploy):
                self.avg = nn.Sequential(OrderedDict([
                    ('avg', nn.AvgPool2d(kernel_size=(self.h, self.w))),
                    ('bn', nn.BatchNorm2d(num_features=C))
                ])
                )
            else:
                self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
            hidden_dim = self.C // self.fc1_fc2_reduction
            self.fc1_fc2 = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(C * self.h_part * self.w_part, hidden_dim)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(hidden_dim, C * self.h_part * self.w_part))
            ])
            )

        self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, kernel_size=1, groups=fc3_groups,
                             bias=self.deploy)
        self.fc3_bn = nn.Identity() if self.deploy else nn.BatchNorm2d(self.O * self.h * self.w)

        if not self.deploy and self.repconv_kernels is not None:
            for k in self.repconv_kernels:
                repconv = nn.Sequential(OrderedDict([
                    ('conv',
                     nn.Conv2d(self.C, self.O, kernel_size=k, padding=(k - 1) // 2, groups=fc3_groups, bias=False)),
                    ('bn', nn.BatchNorm2d(self.O))
                ])

                )
                self.__setattr__('repconv{}'.format(k), repconv)

    def switch_to_deploy(self):
        self.deploy = True
        fc1_weight, fc1_bias, fc3_weight, fc3_bias = self.get_equivalent_fc1_fc3_params()
        # del conv
        if (self.repconv_kernels is not None):
            for k in self.repconv_kernels:
                self.__delattr__('repconv{}'.format(k))
        # del fc3,bn
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, 1, 1, 0, bias=True,
                             groups=self.fc3_groups)
        self.fc3_bn = nn.Identity()
        #   Remove the BN after AVG
        if self.is_global_perceptron:
            self.__delattr__('avg')
            self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
        # Set values
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.data = fc1_weight
            self.fc1_fc2.fc1.bias.data = fc1_bias
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def get_equivalent_fc1_fc3_params(self):
        # training fc3+bn weight
        fc_weight, fc_bias = self._fuse_bn(self.fc3, self.fc3_bn)
        # training conv weight
        if (self.repconv_kernels is not None):
            max_kernel = max(self.repconv_kernels)
            max_branch = self.__getattr__('repconv{}'.format(max_kernel))
            conv_weight, conv_bias = self._fuse_bn(max_branch.conv, max_branch.bn)
            for k in self.repconv_kernels:
                if (k != max_kernel):
                    tmp_branch = self.__getattr__('repconv{}'.format(k))
                    tmp_weight, tmp_bias = self._fuse_bn(tmp_branch.conv, tmp_branch.bn)
                    tmp_weight = F.pad(tmp_weight, [(max_kernel - k) // 2] * 4)
                    conv_weight += tmp_weight
                    conv_bias += tmp_bias
            repconv_weight, repconv_bias = self._conv_to_fc(conv_weight, conv_bias)
            final_fc3_weight = fc_weight + repconv_weight.reshape_as(fc_weight)
            final_fc3_bias = fc_bias + repconv_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias

        # fc1
        if (self.is_global_perceptron):
            # remove BN after avg
            avgbn = self.avg.bn
            std = (avgbn.running_var + avgbn.eps).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn.running_mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.in_features // len(avgbias)
            replicated_avgbias = avgbias.repeat_interleave(replicate_times).view(-1, 1)
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            final_fc1_bias = fc1.bias + bias_diff
            final_fc1_weight = fc1.weight * scale.repeat_interleave(replicate_times).view(1, -1)

        else:
            final_fc1_weight = None
            final_fc1_bias = None

        return final_fc1_weight, final_fc1_bias, final_fc3_weight, final_fc3_bias


    def _conv_to_fc(self, conv_kernel, conv_bias):
        I = torch.eye(self.C * self.h * self.w // self.fc3_groups).repeat(1, self.fc3_groups).reshape(
            self.C * self.h * self.w // self.fc3_groups, self.C, self.h, self.w).to(conv_kernel.device)
        fc_k = F.conv2d(I, conv_kernel, padding=conv_kernel.size(2) // 2, groups=self.fc3_groups)
        fc_k = fc_k.reshape(self.C * self.h * self.w // self.fc3_groups, self.O * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias

    def _fuse_bn(self, conv_or_fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        if conv_or_fc.weight.ndim == 4:
            t = t.reshape(-1, 1, 1, 1)
        else:
            t = t.reshape(-1, 1)
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def forward(self, x):
        ### global partition
        if (self.is_global_perceptron):
            input = x
            v = self.avg(x)  # bs,C,h_part,w_part
            v = v.reshape(-1, self.C * self.h_part * self.w_part)  # bs,C*h_part*w_part
            v = self.fc1_fc2(v)  # bs,C*h_part*w_part
            v = v.reshape(-1, self.C, self.h_part, 1, self.w_part, 1)  # bs,C,h_part,w_part
            input = input.reshape(-1, self.C, self.h_part, self.h, self.w_part, self.w)  # bs,C,h_part,h,w_part,w
            input = v + input
        else:
            input = x.view(-1, self.C, self.h_part, self.h, self.w_part, self.w)  # bs,C,h_part,h,w_part,w
        partition = input.permute(0, 2, 4, 1, 3, 5)  # bs,h_part,w_part,C,h,w

        ### partition partition
        fc3_out = partition.reshape(-1, self.C * self.h * self.w, 1, 1)  # bs*h_part*w_part,C*h*w,1,1
        fc3_out = self.fc3_bn(self.fc3(fc3_out))  # bs*h_part*w_part,O*h*w,1,1
        fc3_out = fc3_out.reshape(-1, self.h_part, self.w_part, self.O, self.h, self.w)  # bs,h_part,w_part,O,h,w

        ### local perceptron
        if (self.repconv_kernels is not None and not self.deploy):
            conv_input = partition.reshape(-1, self.C, self.h, self.w)  # bs*h_part*w_part,C,h,w
            conv_out = 0
            for k in self.repconv_kernels:
                repconv = self.__getattr__('repconv{}'.format(k))
                conv_out += repconv(conv_input)  ##bs*h_part*w_part,O,h,w
            conv_out = conv_out.view(-1, self.h_part, self.w_part, self.O, self.h, self.w)  # bs,h_part,w_part,O,h,w
            fc3_out += conv_out
        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # bs,O,h_part,h,w_part,w
        fc3_out = fc3_out.reshape(-1, self.C, self.H, self.W)  # bs,O,H,W
        return fc3_out


if __name__=='__main__':
    input = torch.randn(50, 3, 14, 14)
    res_mlp = ResMLP(dim=128, image_size=14, patch_size=7, class_num=1000)
    print(res_mlp)
    out = res_mlp(input)
    print(out.shape)


    input = torch.randint(10000, (50, 49))  # bs,len_sen
    gmlp = gMLP(num_tokens=10000, len_sen=49, dim=512, d_ff=1024)
    print(gmlp)
    out = gmlp(input)
    print(out.shape)

    input = torch.randn(50, 3, 40, 40)
    mlp_mixer = MlpMixer(num_classes=1000, num_blocks=10, patch_size=10, tokens_hidden_dim=32, channels_hidden_dim=1024,
                         tokens_mlp_dim=16, channels_mlp_dim=1024)
    print(mlp_mixer)
    output = mlp_mixer(input)
    print(output.shape)

    N = 4  # batch size
    C = 512  # input dim
    O = 1024  # output dim
    H = 14  # image height
    W = 14  # image width
    h = 7  # patch height
    w = 7  # patch width
    fc1_fc2_reduction = 1  # reduction ratio
    fc3_groups = 8  # groups
    repconv_kernels = [1, 3, 5, 7]  # kernel list
    repmlp = RepMLP(C, O, H, W, h, w, fc1_fc2_reduction, fc3_groups, repconv_kernels=repconv_kernels)
    x = torch.randn(N, C, H, W)
    repmlp.eval()
    for module in repmlp.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    # training result
    out = repmlp(x)
    # inference result
    repmlp.switch_to_deploy()
    deployout = repmlp(x)

    print(((deployout - out) ** 2).sum())