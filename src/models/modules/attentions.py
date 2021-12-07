# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/21 18:59
# @Author : liumin
# @File : attentions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

__all__ = ['ExternalAttention', 'SEAttention', 'SKAttention', 'SimAM', 'SAM', 'TripletAttention', 'BAM', 'CBAM', 'NAM','ResidualAttention',
           'FcaAttention', 'ShuffleAttention', 'ECAAttention', 'PyramidSplitAttention', 'SpatialGroupEnhance',
            'ScaledDotProductAttention', 'SimplifiedScaledDotProductAttention', 'EfficientMultiheadSelfAttention', 'CrissCrossAttention',
           'NonLocalAttention', 'GlobalContextAttention', 'DoubleAttention', 'AttentionFreeTransformer', 'MUSEAttention']


class ExternalAttention(nn.Module):
    '''
        Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks
        https://arxiv.org/abs/2105.02358
    '''
    def __init__(self, d_model,S=64):
        super().__init__()
        self.mk=nn.Linear(d_model, S, bias=False)
        self.mv=nn.Linear(S, d_model, bias=False)
        self.softmax=nn.Softmax(dim=1)


    def forward(self, queries):
        attn=self.mk(queries) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2, keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        return out


class SEAttention(nn.Module):
    """
        Squeeze-and-Excitation Networks
        https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channel, reduction=16):
        super(SEAttention, self).__init__()
        mid_channel = channel // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1,3,5,7], reduction=16, group=1, L=32):
        super(SKAttention, self).__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group),
                nn.BatchNorm2d(channel),
                 nn.ReLU()
                )
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w
        ### fuse
        U = sum(conv_outs)  # bs,c,h,w
        ### reduction channel
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d
        ### calculate attention weight
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1
        ### fuse
        V = (attention_weights * feats).sum(0)
        return V


class SimAM(nn.Module):
    '''
        SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
        http://proceedings.mlr.press/v139/yang21o.html
    '''
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


class SAM(nn.Module):
    """SAM layer from `"CBAM: Convolutional Block Attention Module" <https://arxiv.org/pdf/1807.06521.pdf>`_
    modified in `"YOLOv4: Optimal Speed and Accuracy of Object Detection" <https://arxiv.org/pdf/2004.10934.pdf>`_.

    Args:
        in_channels (int): input channels
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.conv(x))


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()

        self.channel_pool = ChannelPool()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(1)
        )
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        out = self.conv(self.channel_pool(x))
        return out * self.sigmod(out)


class TripletAttention(nn.Module):
    def __init__(self, spatial=True):
        super(TripletAttention, self).__init__()
        self.spatial = spatial
        self.height_gate = SpatialGate()
        self.width_gate = SpatialGate()
        if self.spatial:
            self.spatial_gate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.height_gate(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()

        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.width_gate(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()

        if self.spatial:
            x_out3 = self.spatial_gate(x)
            return (1/3) * (x_out1 + x_out2 + x_out3)
        else:
            return (1/2) * (x_out1 + x_out2)

class sSEBlock(nn.Module):
    def __init__(self, channel):
        super(sSEBlock, self).__init__()
        self.spatial_excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.spatial_excitation(x)
        return x * z.expand_as(x)


class cSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(cSEBlock, self).__init__()
        mid_channel = channel // reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.squeeze(x)
        z = self.excitation(y)
        return x * z.expand_as(x)


class scSEBlock(nn.Module):
    """
        Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
        https://arxiv.org/pdf/1803.02579v2.pdf
    """
    def __init__(self, channel, reduction=16):
        super(scSEBlock, self).__init__()
        self.cSE = cSEBlock(channel, reduction)
        self.sSE = sSEBlock(channel)

    def forward(self, x):
        return self.cSE(x) + self.sSE(x)


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    """
        CBAM: Convolutional Block Attention Module
        https://arxiv.org/pdf/1807.06521.pdf
    """
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        mid_channel = channel // reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )

    def forward(self, x):
        avg = self.avg_pool(x).view(x.size(0), -1)
        out = self.shared_MLP(avg).unsqueeze(2).unsqueeze(3).expand_as(x)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, dilation_conv_num=2, dilation_rate=4):
        super(SpatialAttention, self).__init__()
        mid_channel = channel // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channel, mid_channel, kernel_size=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True)
        )
        dilation_convs_list = []
        for i in range(dilation_conv_num):
            dilation_convs_list.append(nn.Conv2d(mid_channel, mid_channel, kernel_size=3, padding=dilation_rate, dilation=dilation_rate))
            dilation_convs_list.append(nn.BatchNorm2d(mid_channel))
            dilation_convs_list.append(nn.ReLU(inplace=True))
        self.dilation_convs = nn.Sequential(*dilation_convs_list)
        self.final_conv = nn.Conv2d(mid_channel, 1, kernel_size=1)

    def forward(self, x):
        y = self.reduce_conv(x)
        x = self.dilation_convs(y)
        out = self.final_conv(y).expand_as(x)
        return out


class BAM(nn.Module):
    """
        BAM: Bottleneck Attention Module
        https://arxiv.org/pdf/1807.06514.pdf
    """
    def __init__(self, channel):
        super(BAM, self).__init__()
        self.channel_attention = ChannelAttention(channel)
        self.spatial_attention = SpatialAttention(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return (1 + att) * x


class NAM(nn.Module):
    """
        NAM: Normalization-based Attention Module
        https://arxiv.org/pdf/2111.12419.pdf
    """
    def __init__(self, channels, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        return x


class ResidualAttention(nn.Module):
    '''
        Residual Attention: A Simple but Effective Method for Multi-Label Recognition
        https://arxiv.org/abs/2108.02456
    '''
    def __init__(self, channel=512 , num_class=1000,la=0.2):
        super().__init__()
        self.la=la
        self.fc=nn.Conv2d(in_channels=channel,out_channels=num_class,kernel_size=1,stride=1,bias=False)

    def forward(self, x):
        b,c,h,w=x.shape
        y_raw=self.fc(x).flatten(2) #b,num_class,hxw
        y_avg=torch.mean(y_raw,dim=2) #b,num_class
        y_max=torch.max(y_raw,dim=2)[0] #b,num_class
        score=y_avg+self.la*y_max
        return score


class FcaAttention(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel, reduction=16, dct_weight=None):
        super(FcaAttention, self).__init__()
        mid_channel = channel // reduction
        self.dct_weight = dct_weight
        self.excitation = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.sum(x*self.dct_weight, dim=[2,3])
        z = self.excitation(y).view(b, c, 1, 1)
        return x * z.expand_as(x)


class ShuffleAttention(nn.Module):
    '''
        SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS
        https://arxiv.org/pdf/2102.00240.pdf
    '''
    def __init__(self, channel=512, reduction=16, G=8):
        super(ShuffleAttention, self).__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w
        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w
        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cweight  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)
        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w
        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


class ECAAttention(nn.Module):
    '''
        ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks
        https://arxiv.org/pdf/1910.03151.pdf
    '''
    def __init__(self, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        y=self.gap(x) #bs,c,1,1
        y=y.squeeze(-1).permute(0,2,1) #bs,1,c
        y=self.conv(y) #bs,1,c
        y=self.sigmoid(y) #bs,1,c
        y=y.permute(0,2,1).unsqueeze(-1) #bs,c,1,1
        return x*y.expand_as(x)


class PyramidSplitAttention(nn.Module):
    '''
        EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network
        https://arxiv.org/pdf/2105.14447.pd
    '''
    def __init__(self, channel=512, reduction=4, S=4):
        super().__init__()
        self.S = S
        self.convs = []
        for i in range(S):
            self.convs.append(nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1))

        self.se_blocks = []
        for i in range(S):
            self.se_blocks.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        # Step1:SPC module
        SPC_out = x.view(b, self.S, c // self.S, h, w)  # bs,s,ci,h,w
        for idx, conv in enumerate(self.convs):
            SPC_out[:, idx, :, :, :] = conv(SPC_out[:, idx, :, :, :])
        # Step2:SE weight
        SE_out = torch.zeros_like(SPC_out)
        for idx, se in enumerate(self.se_blocks):
            SE_out[:, idx, :, :, :] = se(SPC_out[:, idx, :, :, :])
        # Step3:Softmax
        softmax_out = self.softmax(SE_out)
        # Step4:SPA
        PSA_out = SPC_out * softmax_out
        PSA_out = PSA_out.view(b, -1, h, w)
        return PSA_out


class SpatialGroupEnhance(nn.Module):
    '''
        Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks
        https://arxiv.org/pdf/1905.09646.pdf
    '''
    def __init__(self, groups):
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b * self.groups, -1, h, w)  # bs*g,dim//g,h,w
        xn = x * self.avg_pool(x)  # bs*g,dim//g,h,w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g,1,h,w
        t = xn.view(b * self.groups, -1)  # bs*g,h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g,h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g,h*w
        t = t.view(b, self.groups, h, w)  # bs,g,h*w

        t = t * self.weight + self.bias  # bs,g,h*w
        t = t.view(b * self.groups, 1, h, w)  # bs*g,1,h*w
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x


class Depth_Pointwise_Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k):
        super().__init__()
        if (k == 1):
            self.depth_conv = nn.Identity()
        else:
            self.depth_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, groups=1)

    def forward(self, x):
        out = self.pointwise_conv(self.depth_conv(x))
        return out


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """
        CCNet: Criss-Cross Attention for Semantic Segmentation
        https://arxiv.org/pdf/1811.11721.pdf
    """
    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)

        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        return self.gamma * (out_H + out_W) + x



class NonLocalAttention(nn.Module):
    """
        Non-local Neural Networks
        https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, channel):
        super(NonLocalAttention, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                  bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                   bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class GlobalContextAttention(nn.Module):
    """
        GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
        https://arxiv.org/pdf/1904.11492.pdf
    """
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add', )):
        super(GlobalContextAttention, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes // ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class ScaledDotProductAttention(nn.Module):
    '''
        Attention Is All You Need
        https://arxiv.org/pdf/1706.03762.pdf
    '''
    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model//h
        self.d_v = d_model//h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class EfficientMultiheadSelfAttention(nn.Module):
    '''
        ResT: An Efficient Transformer for Visual Recognition
        https://arxiv.org/abs/2105.13677
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):
        super(EfficientMultiheadSelfAttention, self).__init__()
        self.H = H
        self.W = W
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio
        if (self.ratio > 1):
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2,
                                     groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):

        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            x = self.sr_conv(x)  # bs,c,h,w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class DoubleAttention(nn.Module):
    '''
        A2-Nets: Double Attention Networks
        https://arxiv.org/pdf/1810.11579.pdf
    '''
    def __init__(self, in_channels,c_m,c_n,reconstruct = True):
        super().__init__()
        self.in_channels=in_channels
        self.reconstruct = reconstruct
        self.c_m=c_m
        self.c_n=c_n
        self.convA=nn.Conv2d(in_channels,c_m,1)
        self.convB=nn.Conv2d(in_channels,c_n,1)
        self.convV=nn.Conv2d(in_channels,c_n,1)
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size = 1)

    def forward(self, x):
        b, c, h,w=x.shape
        assert c==self.in_channels
        A=self.convA(x) #b,c_m,h,w
        B=self.convB(x) #b,c_n,h,w
        V=self.convV(x) #b,c_n,h,w
        tmpA=A.view(b,self.c_m,-1)
        attention_maps=F.softmax(B.view(b,self.c_n,-1))
        attention_vectors=F.softmax(V.view(b,self.c_n,-1))
        # step 1: feature gating
        global_descriptors=torch.bmm(tmpA,attention_maps.permute(0,2,1)) #b.c_m,c_n
        # step 2: feature distribution
        tmpZ = global_descriptors.matmul(attention_vectors) #b,c_m,h*w
        tmpZ=tmpZ.view(b,self.c_m,h,w) #b,c_m,h,w
        if self.reconstruct:
            tmpZ=self.conv_reconstruct(tmpZ)
        return tmpZ


class AttentionFreeTransformer(nn.Module):
    '''
        An Attention Free Transformer
        https://arxiv.org/pdf/2105.14103v1.pdf
    '''
    def __init__(self, d_model, n=49, simple=False):
        super(AttentionFreeTransformer, self).__init__()
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        if (simple):
            self.position_biases = torch.zeros((n, n))
        else:
            self.position_biases = nn.Parameter(torch.ones((n, n)))
        self.d_model = d_model
        self.n = n
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        bs, n, dim = input.shape

        q = self.fc_q(input)  # bs,n,dim
        k = self.fc_k(input).view(1, bs, n, dim)  # 1,bs,n,dim
        v = self.fc_v(input).view(1, bs, n, dim)  # 1,bs,n,dim

        numerator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)) * v, dim=2)  # n,bs,dim
        denominator = torch.sum(torch.exp(k + self.position_biases.view(n, 1, -1, 1)), dim=2)  # n,bs,dim

        out = (numerator / denominator)  # n,bs,dim
        out = self.sigmoid(q) * (out.permute(1, 0, 2))  # bs,n,dim
        return out


class MUSEAttention(nn.Module):
    '''
        MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning
        https://arxiv.org/abs/1911.09483
    '''
    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        super(MUSEAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.conv1 = Depth_Pointwise_Conv1d(h * d_v, d_model, 1)
        self.conv3 = Depth_Pointwise_Conv1d(h * d_v, d_model, 3)
        self.conv5 = Depth_Pointwise_Conv1d(h * d_v, d_model, 5)
        self.dy_paras = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(-1)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        # Self Attention
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        v2 = v.permute(0, 1, 3, 2).contiguous().view(b_s, -1, nk)  # bs,dim,n
        self.dy_paras = nn.Parameter(self.softmax(self.dy_paras))
        out2 = self.dy_paras[0] * self.conv1(v2) + self.dy_paras[1] * self.conv3(v2) + self.dy_paras[2] * self.conv5(v2)
        out2 = out2.permute(0, 2, 1)  # bs.n.dim

        out = out + out2
        return out



if __name__=='__main__':
    # model = SEBlock(16)
    # model = scSEBlock(16, 4)
    # model = NonLocalBlock(16)
    # model = GlobalContextBlock(inplanes=16, ratio=4, pooling_type='att')
    # model = CBAM(16)
    model = BAM(16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)