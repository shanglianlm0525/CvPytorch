# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/12/14 16:38
# @Author : liumin
# @File : incep_transformer.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import DropPath, build_norm_layer
from src.models.init.weight_init import trunc_normal_, trunc_normal_init, constant_init, normal_init


class DWConv(nn.Module):
    def __init__(self, embed_dim=768, kernel_size=3, padding=2):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size, stride=1, padding=padding, bias=True,
                                groups=embed_dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features, 3, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # x: B C H W
    def forward(self, x, H, W):
        x = self.act(self.fc1(x))
        x = self.act(self.dwconv(x))
        x = self.act(self.fc2(x))
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, down_ratio=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0., ):
        super().__init__()
        self.down_ratio = down_ratio
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        if down_ratio > 1:
            self.conv1 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(1, down_ratio), stride=(1, down_ratio), groups=embed_dim),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=(down_ratio, 1), stride=(down_ratio, 1), groups=embed_dim),
                # norm_layer(embed_dim),
                # act_layer()
            )

            self.conv2 = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=down_ratio, stride=down_ratio, groups=embed_dim),
                # nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)
            )

            self.dwConv = DWConv(embed_dim, 3, 1)
            self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            self.kv = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # B C H W
        B, C, _, _ = x.shape
        N = H * W
        x_layer = x.reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(x_layer).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.down_ratio > 1:
            x_1 = self.conv1(x).view(B, C, -1)
            x_2 = self.conv2(x).view(B, C, -1)
            x_3 = F.adaptive_avg_pool2d(x, (H // self.down_ratio, W // self.down_ratio))
            x_3 = self.dwConv(x_3).view(B, C, -1)
            x_ = torch.cat([x_1, x_2, x_3], dim=2)
            x_ = self.norm(x_.permute(0, 2, 1))
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x_layer).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, down_ratio=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 proj_drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=dict(type='BN', requires_grad=True), drop=0.):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, embed_dim)[1]
        self.attn = Attention(
            embed_dim, num_heads,
            down_ratio=down_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, embed_dim)[1]
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        short_cut = x
        B, C, _, _ = x.shape
        x = self.drop_path(self.attn(self.norm1(x), H, W)).permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        x = x + short_cut
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        patch_size = (patch_size, patch_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class IncepTransformer(nn.Module):
    cfg = {"t": ([64, 128, 320, 512], [2, 2, 4, 2], [2, 4, 8, 16], [8, 8, 4, 4], [8, 4, 2, 1]),
           "s": ([64, 128, 320, 512], [3, 4, 12, 2], [2, 4, 8, 16], [8, 8, 4, 4], [8, 4, 2, 1]),
           "b": ([64, 128, 320, 512], [3, 6, 24, 2], [2, 4, 8, 16], [8, 8, 4, 4], [8, 4, 2, 1])}
    def __init__(self, subtype='ipt_t', out_channels=[64, 128, 320, 512], depths=[3, 6, 24, 2], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=True,
                 down_ratios=[8, 4, 2, 1], drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_cfg=dict(type='BN', requires_grad=True),
                 out_stages=[1, 2, 3, 4], output_stride = 32, classifier=False, backbone_path=None, pretrained = False):
        super(IncepTransformer, self).__init__()
        self.subtype = subtype
        self.out_channels = out_channels
        self.out_stages = out_stages
        self.output_stride = output_stride # 8, 16, 32
        self.classifier = classifier
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(len(self.out_channels)):
            patch_embed = OverlapPatchEmbed(patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=3 if i == 0 else self.out_channels[i - 1],
                                            embed_dim=self.out_channels[i],
                                            norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                embed_dim=self.out_channels[i], num_heads=num_heads[i], down_ratio=down_ratios[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path=dpr[cur + j], norm_cfg=norm_cfg, drop=drop_rate)
                for j in range(depths[i])])
            norm = nn.BatchNorm2d(self.out_channels[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # self.out_channels = [self.out_channels[ost-1] for ost in self.out_stages]

        if self.pretrained and self.backbone_path is not None:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def forward(self, x):
        B = x.shape[0]
        output = []

        for i in range(4):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i + 1 in self.out_stages:
                output.append(x)

        return output


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0.)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[
                    1] * m.out_channels
                fan_out //= m.groups
                normal_init(m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            state_dict = torch.load(self.backbone_path)["state_dict"]
        else:
            raise NotImplementedError
        model_state_dict = self.state_dict()
        for (k1, v1), (k2, v2) in zip(model_state_dict.items(), state_dict.items()):
            if v1.shape == v2.shape:
                model_state_dict.update({k1: v2})
        self.load_state_dict(model_state_dict)


if __name__ == "__main__":
    model = IncepTransformer('ipt_b')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)