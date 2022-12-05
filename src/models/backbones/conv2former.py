# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/12/2 17:44
# @Author : liumin
# @File : conv2former.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.bricks import DropPath


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))
        x = self.fc2(x)
        return x


class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.a(x) * self.v(x)
        x = self.proj(x)
        return x


class ConvModBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop_path=0.):
        super().__init__()

        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class Conv2Former(nn.Module):
    cfg = {
        'n': [64, 128, 256, 512],
        't': [72, 144, 288, 576],
        's': [72, 144, 288, 576],
        'b': [96, 192, 384, 768],
        'l': [128, 256, 512, 1024],
    }
    def __init__(self, subtype='conv2former_b', out_stages=[2, 3, 4], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):
        super(Conv2Former, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        if self.subtype == 'conv2former_n':
            convnext = models.convnext_tiny(pretrained=self.pretrained)
            self.out_channels = [96, 96, 192, 384, 768]
        elif self.subtype == 'conv2former_t':
            convnext = models.convnext_small(pretrained=self.pretrained)
            self.out_channels = [96, 96, 192, 384, 768]
        elif self.subtype == 'conv2former_s':
            convnext = models.convnext_base(pretrained=self.pretrained)
            self.out_channels = [128, 128, 256, 512, 1024]
        elif self.subtype == 'conv2former_b':
            convnext = models.convnext_large(pretrained=self.pretrained)
            self.out_channels = [256, 256, 512, 1024, 2048]
        elif self.subtype == 'conv2former_l':
            convnext = models.convnext_large(pretrained=self.pretrained)
            self.out_channels = [256, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.patch_embedding1 = nn.Conv2d(1, 1, 2, 2)
        self.stage1 = self._make_stage(block, 64, layers[0])

        self.patch_embedding2 = nn.Conv2d(1, 1, 2, 2)

        self.patch_embedding3 = nn.Conv2d(1, 1, 2, 2)

        self.patch_embedding4 = nn.Conv2d(1, 1, 2, 2)

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        features = list(convnext.features.children())
        self.stem = features[0] # x2
        self.stage1 = features[1]
        self.stage2 = nn.Sequential(*features[2:4])
        self.stage3 = nn.Sequential(*features[4:6])
        self.stage4 = nn.Sequential(*features[6:])
        if self.classifier:
            self.avgpool = convnext.avgpool
            self.fc = convnext.classifier
            self.fc[2] = nn.Linear(self.fc[2].in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def _make_layer(self, inplanes, planes, blocks):
        layers = []
        layers.append(nn.Conv2d(1, 1, 2, 2))
        for _ in range(blocks):
            layers.append(ConvModBlock(dim, mlp_ratio=4., drop_path=0.))
        return nn.Sequential(*layers)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))

    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.avgpool(x)
            x = self.fc(x)
            return x
        return output if len(self.out_stages) > 1 else output[0]



if __name__=='__main__':
    model = Conv2Former('conv2former_b', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    if isinstance(out, list):
        for o in out:
            print(o.shape)
    else:
        print(out.shape)