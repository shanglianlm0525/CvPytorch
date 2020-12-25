# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/12/21 18:59
# @Author : liumin
# @File : attentions.py

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
        Squeeze-and-Excitation Networks
        https://arxiv.org/pdf/1709.01507.pdf
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
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


class SKBlock(nn.Module):
    def __init__(self):
        super(SKBlock, self).__init__()

    def forward(self, x):
        pass


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


class NonLocalBlock(nn.Module):
    """
        Non-local Neural Networks
        https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
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


class GlobalContextBlock(nn.Module):
    """
        GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond
        https://arxiv.org/pdf/1904.11492.pdf
    """
    def __init__(self, inplanes, ratio, pooling_type='att', fusion_types=('channel_add', )):
        super(GlobalContextBlock, self).__init__()
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


class CrissCrossAttention(nn.Module):
    def __init__(self):
        super(CrissCrossAttention, self).__init__()

    def forward(self, x):
        pass


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
        att = 1 + self.sigmoid(self.channel_attention(x) * self.spatial_attention(x))
        return att * x


class SplitAttention(nn.Module):
    def __init__(self):
        super(SplitAttention, self).__init__()

    def forward(self, x):
        pass

'''
class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """
    def __init__(self, in_channels, channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True,
                 radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, norm_layer=None,
                 dropblock_prob=0.0, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        self.rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        self.rectify_avg = rectify_avg
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = channels
        self.dropblock_prob = dropblock_prob
        if self.rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                                 groups=groups*radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, channels*radix, kernel_size, stride, padding, dilation,
                               groups=groups*radix, bias=bias, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn0 = norm_layer(channels*radix)
        self.relu = ReLU(inplace=True)
        self.fc1 = Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(inter_channels)
        self.fc2 = Conv2d(inter_channels, channels*radix, 1, groups=self.cardinality)
        if dropblock_prob > 0.0:
            self.dropblock = DropBlock2D(dropblock_prob, 3)
        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn0(x)
        if self.dropblock_prob > 0.0:
            x = self.dropblock(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel//self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.use_bn:
            gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()
'''

class FCABlock(nn.Module):
    """
        FcaNet: Frequency Channel Attention Networks
        https://arxiv.org/pdf/2012.11879.pdf
    """
    def __init__(self, channel, reduction=16, dct_weight=None):
        super(FCABlock, self).__init__()
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


if __name__=='__main__':
    # model = SEBlock(16)
    # model = scSEBlock(16, 4)
    # model = NonLocalBlock(16)
    # model = GlobalContextBlock(inplanes=16, ratio=4, pooling_type='att')
    # model = CBAM(16)
    # model = BAM(16)
    model = FCABlock(16)
    print(model)

    input = torch.randn(1, 16, 64, 64)
    out = model(input)
    print(out.shape)