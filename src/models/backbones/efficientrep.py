# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/6/24 13:41
# @Author : liumin
# @File : efficientrep.py

import math
import torch
import torch.nn as nn

from src.models.modules.yolov6_modules import RepVGGBlock, RepBlock, SimSPPF


class EfficientRep(nn.Module):
    cfg = {"nano": [0.33, 0.25],
            "tiny": [0.33, 0.375],
            "s": [0.33, 0.5],
            "m": [0.67, 0.75],
            "l": [1.0, 1.0],
            "x": [1.33, 1.25]}
    def __init__(self, subtype='yolov6_s', out_stages=[2, 3, 4], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):
        super(EfficientRep, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        depth_mul, width_mul = self.cfg[self.subtype.split("_")[1]]
        layers = [6, 12, 18, 6]
        out_channels = [64, 128, 256, 512, 1024]
        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))
        self.out_channels = in_places = list(map(lambda x: int(x * width_mul), out_channels))

        self.stem = RepVGGBlock(in_channels=3, out_channels=in_places[0], kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
                RepVGGBlock(in_channels=in_places[0], out_channels=in_places[1], kernel_size=3, stride=2),
                RepBlock(in_channels=in_places[1], out_channels=in_places[1], n=layers[0])
            )
        self.stage2 = nn.Sequential(
                RepVGGBlock(in_channels=in_places[1], out_channels=in_places[2], kernel_size=3, stride=2),
                RepBlock(in_channels=in_places[2], out_channels=in_places[2], n=layers[1])
            )
        self.stage3 = nn.Sequential(
                RepVGGBlock(in_channels=in_places[2], out_channels=in_places[3], kernel_size=3, stride=2),
                RepBlock(in_channels=in_places[3], out_channels=in_places[3], n=layers[2])
            )
        self.stage4 = nn.Sequential(
                RepVGGBlock(in_channels=in_places[3], out_channels=in_places[4], kernel_size=3, stride=2),
                RepBlock(in_channels=in_places[4],out_channels=in_places[4], n=layers[3]),
                SimSPPF(in_channels=in_places[4], out_channels=in_places[4], kernel_size=5)
            )

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)


    def forward(self, x):
        x = self.stem(x)
        output = []
        for i in range(1, 5):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        return output if len(self.out_stages) > 1 else output[0]

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__=='__main__':
    model = EfficientRep('yolov6_s')
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    if isinstance(out, list):
        for o in out:
            print(o.shape)
    else:
        print(out.shape)
