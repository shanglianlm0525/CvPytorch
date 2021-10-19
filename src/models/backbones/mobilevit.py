# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/10/15 16:59
# @Author : liumin
# @File : MobileViT.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def Conv3x3BN(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )


def Conv3x3BNActivation(in_channels,out_channels,stride=1,groups=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )


def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )

def Conv1x1BNActivation(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )


class MV2Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor=6):
        super(MV2Block, self).__init__()
        self.stride = stride
        mid_channels = (in_channels * expansion_factor)

        self.bottleneck = nn.Sequential(
            Conv1x1BNActivation(in_channels, mid_channels),
            Conv3x3BNActivation(mid_channels, mid_channels, stride, groups=mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )

        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)

    def forward(self, x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = Conv3x3BNActivation(channel, channel)
        self.conv2 = Conv1x1BNActivation(channel, dim)

        self.transformer = Transformer(dim, depth, 1, 32, mlp_dim, dropout)

        self.conv3 = Conv1x1BNActivation(dim, channel)
        self.conv4 = Conv3x3BNActivation(2 * channel, channel)

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h // self.ph, w=w // self.pw, ph=self.ph,
                      pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Module):
    # def __init__(self, dims, channels, expansion=4, patch_size=(2, 2), num_classes=1000):
    def __init__(self, subtype='mobilevit_s', out_stages=[3, 4, 5], output_stride=16, classifier=False, backbone_path=None, pretrained = False):
        super(MobileViT, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        patch_size = (2, 2)
        if self.subtype == 'mobilevit_xxs':
            dims = [64, 80, 96]
            channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
            expansion = 2
        elif self.subtype == 'mobilevit_xs':
            dims = [96, 120, 144]
            channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
            expansion = 1
        elif self.subtype == 'mobilevit_s':
            dims = [144, 192, 240]
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
            expansion = 1
        else:
            raise NotImplementedError


        self.out_channels = [32, 16, 24, 32, 64, 96, 160, 320]

        depth = [2, 4, 3]

        self.conv1 = Conv3x3BNActivation(3, channels[0], 2)
        self.layer1 = MV2Block(in_channels=channels[0], out_channels=channels[1], stride=1, expansion_factor=expansion)

        self.layer2 = nn.Sequential(
            MV2Block(in_channels=channels[1], out_channels=channels[2], stride=2, expansion_factor=expansion),
            MV2Block(in_channels=channels[2], out_channels=channels[3], stride=1, expansion_factor=expansion),
            MV2Block(in_channels=channels[3], out_channels=channels[3], stride=1, expansion_factor=expansion)
        )

        self.layer3 = nn.Sequential(
            MV2Block(in_channels=channels[3], out_channels=channels[4], stride=2, expansion_factor=expansion),
            MobileViTBlock(dim=dims[0], depth=depth[0], channel=channels[5], patch_size=patch_size, mlp_dim=int(dims[0]*2) )
        )

        self.layer4 = nn.Sequential(
            MV2Block(in_channels=channels[5], out_channels=channels[6], stride=2, expansion_factor=expansion),
            MobileViTBlock(dim=dims[1], depth=depth[1], channel=channels[7], patch_size=patch_size, mlp_dim=int(dims[1]*4) )
        )

        self.layer5 = nn.Sequential(
            MV2Block(in_channels=channels[7], out_channels=channels[8], stride=2, expansion_factor=expansion),
            MobileViTBlock(dim=dims[2], depth=depth[2], channel=channels[9], patch_size=patch_size, mlp_dim=int(dims[2]*4) )
        )

        if self.classifier:
            self.last_conv = Conv1x1BNActivation(channels[9], channels[10])
            self.avgpool = nn.AvgPool2d(kernel_size=8, stride=1)
            self.dropout = nn.Dropout(p=0.2)
            self.linear = nn.Linear(in_features=channels[10], out_features=1000)
            self.out_channels = [1000]

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        output = []
        for i in range(1, 6):
            stage = getattr(self, 'layer{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.last_conv(x)
            x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.dropout(x)
            x = self.linear(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]


if __name__=="__main__":
    model =MobileViT('mobilevit_s')
    print(model)

    input = torch.randn(1, 3, 256, 256)
    out = model(input)
    for o in out:
        print(o.shape)