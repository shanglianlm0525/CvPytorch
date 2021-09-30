# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/23 14:27
# @Author : liumin
# @File : openpose_head.py

import torch
import torch.nn as nn

'''
    OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields
    https://arxiv.org/pdf/1812.08008.pdf
'''

class OpenPoseHead(nn.Module):
    def __init__(self, num_classes=19, in_channels=128):
        super(OpenPoseHead, self).__init__()
        mid_channels = in_channels + num_classes + 2 * num_classes

        # Stage 1
        self.model1_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels*4, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, num_classes*2, kernel_size=(1, 1), stride=(1, 1))
        )
        self.model1_2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels*4, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels*4, num_classes, kernel_size=(1, 1), stride=(1, 1))
        )

        # Stages 2 - 6
        self.modelx_1_list = nn.ModuleList()
        self.modelx_2_list = nn.ModuleList()
        for i in range(5):
            self.modelx_1_list.append(nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_classes*2, kernel_size=(1, 1), stride=(1, 1))
            ))

            self.modelx_2_list.append(nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1), stride=(1, 1))
            ))

        self._init_weight()

    def forward(self, x):
        vecs = []
        heats = []
        out_1 = self.model1_1(x)
        out_2 = self.model1_2(x)
        vecs.append(out_1)
        heats.append(out_2)
        for modelx_1, modelx_2 in zip(self.modelx_1_list, self.modelx_2_list):
            out = torch.cat([out_1, out_2, x], 1)
            out_1 = modelx_1(out)
            out_2 = modelx_2(out)
            vecs.append(out_1)
            heats.append(out_2)
        return (out_1, out_2), (heats, vecs)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    nn.init.constant_(m.bias, 0.0)

        # last layer of these block don't have Relu
        nn.init.normal_(self.model1_1[-1].weight, std=0.01)
        nn.init.normal_(self.model1_2[-1].weight, std=0.01)

        for modelx_1, modelx_2 in zip(self.modelx_1_list, self.modelx_2_list):
            nn.init.normal_(modelx_1[-1].weight, std=0.01)
            nn.init.normal_(modelx_2[-1].weight, std=0.01)
