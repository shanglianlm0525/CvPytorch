# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/7/6 9:53
# @Author : liumin
# @File : stdcnet.py
import math

import torch
import torch.nn as nn

"""
    Rethinking BiSeNet For Real-time Semantic Segmentation
    https://arxiv.org/pdf/2104.13188.pdf
"""



class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2, padding=1, groups=out_planes // 2,
                          bias=False),
                nn.BatchNorm2d(out_planes // 2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out


class STDCNet(nn.Module):

    def __init__(self, subtype='stdc1', out_channels = [32, 64, 256, 512, 1024], layers = [2, 2, 2], block_num = 4, out_stages=[2, 3, 4], output_stride = 32, classifier=False, num_classes=1000, backbone_path=None, pretrained = False):
        super(STDCNet, self).__init__()
        self.subtype = subtype
        self.out_channels = out_channels
        self.out_stages = out_stages
        self.output_stride = output_stride # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        self.stem = ConvX(3, self.out_channels[0], 3, 2)  # x2
        self.layer1 = ConvX(self.out_channels[0], self.out_channels[1], 3, 2)  # x4
        self.layer2 = self._make_layers(self.out_channels[1], self.out_channels[2], layers[0], block_num,
                                        CatBottleneck)  # AddBottleneck  # x8
        self.layer3 = self._make_layers(self.out_channels[2], self.out_channels[3], layers[1], block_num,
                                        CatBottleneck)  # AddBottleneck # x16
        self.layer4 = self._make_layers(self.out_channels[3], self.out_channels[4], layers[2], block_num,
                                        CatBottleneck)  # AddBottleneck # x32

        if self.classifier:
            self.conv_last = ConvX(self.out_channels[-1], 1024, 1, 1)
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(1024, 1024, bias=False)
            self.bn = nn.BatchNorm1d(1024)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.2)
            self.linear = nn.Linear(1024, self.num_classes, bias=False)

        self.out_channels = [self.out_channels[ost] for ost in self.out_stages]

        if self.pretrained and self.backbone_path is not None:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def _make_layers(self, inplanes, planes, layer, block_num, block):
        features = [block(inplanes, planes, block_num, 2)]
        for j in range(layer-1):
            features.append(block(planes, planes, block_num, 1))
        return nn.Sequential(*features)

    def forward(self, x):
        output = []
        x = self.stem(x)
        for i in range(1, 5):
            layer = getattr(self, 'layer{}'.format(i))
            x = layer(x)
            if i in self.out_stages:
                output.append(x)

        if self.classifier:
            x = self.conv_last(x).pow(2)
            x = self.gap(x).flatten(1)
            x = self.fc(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.linear(x)
            return x

        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def freeze_stages(self, stage):
        if stage >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
        for i in range(1, stage + 1):
            layer = getattr(self, 'layer{}'.format(i))
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def load_pretrained_weights(self):
        if self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            state_dict = torch.load(self.backbone_path)["state_dict"]
        else:
            raise ValueError
        model_state_dict = self.state_dict()
        for (k1, v1), (k2, v2) in zip(model_state_dict.items(), state_dict.items()):
            if v1.shape == v2.shape:
                model_state_dict.update({k1: v2})
        self.load_state_dict(model_state_dict)


if __name__ == "__main__":
    # model = STDCNet('stdc1')
    model = STDCNet('stdc1', backbone_path = "/home/lmin/pythonCode/CvPytorch/weights/stdc/STDCNet813M_73.91.tar", pretrained = True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)