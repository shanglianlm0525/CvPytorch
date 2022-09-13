# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/9/13 16:58
# @Author : liumin
# @File : yolov6_neck.py

import torch
from torch import nn
from src.models.modules.yolov6_modules import RepBlock, Transpose, Conv, CSPStackRep


class YOLOv6Neck(nn.Module):
    """RepPANNeck Module
    EfficientRep is the default backbone of this model.
    RepPANNeck has the balance of feature fusion ability and hardware efficiency.
    """
    def __init__(self, subtype='yolov6_s', in_channels=[256, 512, 1024], mid_channels = [128, 128, 256], out_channels=[128, 256, 512], layers=[12, 12, 12, 12],
                 depth_mul=1.0, width_mul=1.0, csp_e=None):
        super().__init__()
        self.subtype = subtype
        assert in_channels is not None
        assert layers is not None

        layers = list(map(lambda x: max(round(x * depth_mul), 1), layers))
        in_channels = list(map(lambda x: int(x * width_mul), in_channels))
        out_channels = list(map(lambda x: int(x * width_mul), out_channels))
        mid_channels = list(map(lambda x: int(x * width_mul), mid_channels))

        block = CSPStackRep if depth_mul > 0.5 else RepBlock
        self.reduce_layer0 = Conv(in_channels=in_channels[2], out_channels=mid_channels[2], kernel_size=1, stride=1, act=False)
        self.upsample0 = Transpose(in_channels=mid_channels[2], out_channels=mid_channels[2])
        self.Rep_p4 = block(in_channels=in_channels[1] + mid_channels[2], out_channels=mid_channels[2], n=layers[0], e=csp_e)

        self.reduce_layer1 = Conv(in_channels=mid_channels[2], out_channels=mid_channels[1], kernel_size=1, stride=1, act=False)
        self.upsample1 = Transpose(in_channels=mid_channels[1], out_channels=mid_channels[1])
        self.Rep_p3 = block(in_channels=in_channels[0] + mid_channels[1], out_channels=out_channels[0], n=layers[1], e=csp_e)

        self.downsample2 = Conv(in_channels=out_channels[0], out_channels=mid_channels[0], kernel_size=3, stride=2, act=False)
        self.Rep_n3 = block(in_channels=mid_channels[1] + mid_channels[0], out_channels=out_channels[1], n=layers[2], e=csp_e)

        self.downsample1 = Conv(in_channels=out_channels[1], out_channels=mid_channels[2], kernel_size=3, stride=2, act=False)
        self.Rep_n4 = block(in_channels=mid_channels[2] + mid_channels[2], out_channels=out_channels[2], n=layers[3], e=csp_e)


    def forward(self, input):
        x2, x1, x0 = input

        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([upsample_feat0, x1], 1)
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([upsample_feat1, x2], 1)
        pan_out2 = self.Rep_p3(f_concat_layer1)

        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = torch.cat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = torch.cat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs
