# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/6 17:52
# @Author : liumin
# @File : custom_model.py

import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models.mobilenet import mobilenet_v2

from compressions.quantization.quant_ops import QuantConv, PassThroughOp, QuantLinear, Quantizers, LSQActivations

"""
    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and not any(x in k for x in exclude) and v.shape == db[k].shape}


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = nn.Conv2d(conv.in_channels,
                          conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          groups=conv.groups,
                          bias=True).requires_grad_(False).to(conv.weight.device)

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv

def fusebn(model):
    print('Fusing layers... ')
    modules_to_fuse = [['conv1.0.0', 'conv1.0.1'],

                       ['stage1.0.conv.0.0', 'stage1.0.conv.0.1'],
                       ['stage1.0.conv.1', 'stage1.0.conv.2'],

                       ['stage2.0.conv.0.0', 'stage2.0.conv.0.1'],
                       ['stage2.0.conv.1.0', 'stage2.0.conv.1.1'],
                       ['stage2.0.conv.2', 'stage2.0.conv.3'],
                       ['stage2.1.conv.0.0', 'stage2.1.conv.0.1'],
                       ['stage2.1.conv.1.0', 'stage2.1.conv.1.1'],
                       ['stage2.1.conv.2', 'stage2.1.conv.3'],

                       ['stage3.0.conv.0.0', 'stage3.0.conv.0.1'],
                       ['stage3.0.conv.1.0', 'stage3.0.conv.1.1'],
                       ['stage3.0.conv.2', 'stage3.0.conv.3'],
                       ['stage3.1.conv.0.0', 'stage3.1.conv.0.1'],
                       ['stage3.1.conv.1.0', 'stage3.1.conv.1.1'],
                       ['stage3.1.conv.2', 'stage3.1.conv.3'],
                       ['stage3.2.conv.0.0', 'stage3.2.conv.0.1'],
                       ['stage3.2.conv.1.0', 'stage3.2.conv.1.1'],
                       ['stage3.2.conv.2', 'stage3.2.conv.3'],

                       ['stage4.0.conv.0.0', 'stage4.0.conv.0.1'],
                       ['stage4.0.conv.1.0', 'stage4.0.conv.1.1'],
                       ['stage4.0.conv.2', 'stage4.0.conv.3'],
                       ['stage4.1.conv.0.0', 'stage4.1.conv.0.1'],
                       ['stage4.1.conv.1.0', 'stage4.1.conv.1.1'],
                       ['stage4.1.conv.2', 'stage4.1.conv.3'],
                       ['stage4.2.conv.0.0', 'stage4.2.conv.0.1'],
                       ['stage4.2.conv.1.0', 'stage4.2.conv.1.1'],
                       ['stage4.2.conv.2', 'stage4.2.conv.3'],
                       ['stage4.3.conv.0.0', 'stage4.3.conv.0.1'],
                       ['stage4.3.conv.1.0', 'stage4.3.conv.1.1'],
                       ['stage4.3.conv.2', 'stage4.3.conv.3'],

                       ['stage5.0.conv.0.0', 'stage5.0.conv.0.1'],
                       ['stage5.0.conv.1.0', 'stage5.0.conv.1.1'],
                       ['stage5.0.conv.2', 'stage5.0.conv.3'],
                       ['stage5.1.conv.0.0', 'stage5.1.conv.0.1'],
                       ['stage5.1.conv.1.0', 'stage5.1.conv.1.1'],
                       ['stage5.1.conv.2', 'stage5.1.conv.3'],
                       ['stage5.2.conv.0.0', 'stage5.2.conv.0.1'],
                       ['stage5.2.conv.1.0', 'stage5.2.conv.1.1'],
                       ['stage5.2.conv.2', 'stage5.2.conv.3'],

                       ['stage6.0.conv.0.0', 'stage6.0.conv.0.1'],
                       ['stage6.0.conv.1.0', 'stage6.0.conv.1.1'],
                       ['stage6.0.conv.2', 'stage6.0.conv.3'],
                       ['stage6.1.conv.1.0', 'stage6.1.conv.1.1'],
                       ['stage6.1.conv.2', 'stage6.1.conv.3'],
                       ['stage6.2.conv.0.0', 'stage6.2.conv.0.1'],
                       ['stage6.2.conv.1.0', 'stage6.2.conv.1.1'],
                       ['stage6.2.conv.2', 'stage6.2.conv.3'],

                       ['stage7.0.conv.0.0', 'stage7.0.conv.0.1'],
                       ['stage7.0.conv.1.0', 'stage7.0.conv.1.1'],
                       ['stage7.0.conv.2', 'stage7.0.conv.3'],

                       ['last_conv.0.0', 'last_conv.0.1']]

    return torch.quantization.fuse_modules(model, modules_to_fuse)


def replace_quant_ops_new(model, num_bits, quant_scheme):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QuantConv(quant_scheme, child)
            setattr(model, child_name, new_op)
        elif isinstance(child, torch.nn.Linear):
            new_op = QuantLinear(quant_scheme, child)
            setattr(model, child_name, new_op)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            # prev_module.activation_function = child
            setattr(model, child_name, PassThroughOp())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, PassThroughOp())
        else:
            replace_quant_ops(child, num_bits, quant_scheme)
    return model


def replace_quant_ops(model, num_bits, quant_scheme):
    prev_module = None
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.Conv2d):
            new_op = QuantConv(quant_scheme, child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, torch.nn.Linear):
            new_op = QuantLinear(quant_scheme, child)
            setattr(model, child_name, new_op)
            prev_module = getattr(model, child_name)
        elif isinstance(child, (torch.nn.ReLU, torch.nn.ReLU6)):
            # prev_module.activation_function = child
            prev_module.activation_function = torch.nn.ReLU()
            setattr(model, child_name, PassThroughOp())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(model, child_name, PassThroughOp())
        else:
            replace_quant_ops(child, num_bits, quant_scheme)

def run_calibration(calibration):
    def estimate_range(module):
        if isinstance(module, Quantizers):
            module.estimate_range(flag = calibration)
    return estimate_range

def set_quant_mode(quantized):
    def set_precision_mode(module):
        if isinstance(module, (Quantizers, LSQActivations)):
            module.set_quantize(quantized)
            module.estimate_range(flag = False)
    return set_precision_mode

class MobileNetV2(nn.Module):

    def __init__(self, subtype='mobilenet_v2', out_stages=[3, 5, 7], output_stride=16, classifier=False, backbone_path=None, pretrained = False):
        super(MobileNetV2, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.backbone_path = backbone_path
        self.pretrained = pretrained

        if self.subtype == 'mobilenet_v2':
            features = mobilenet_v2(self.pretrained).features
            self.out_channels = [32, 16, 24, 32, 64, 96, 160, 320]
        else:
            raise NotImplementedError

        self.out_channels = self.out_channels[self.out_stages[0]:self.out_stages[-1] + 1]

        self.conv1 = nn.Sequential(list(features.children())[0])
        self.stage1 = nn.Sequential(list(features.children())[1])
        self.stage2 = nn.Sequential(*list(features.children())[2:4])
        self.stage3 = nn.Sequential(*list(features.children())[4:7])
        self.stage4 = nn.Sequential(*list(features.children())[7:11])
        self.stage5 = nn.Sequential(*list(features.children())[11:14])
        self.stage6 = nn.Sequential(*list(features.children())[14:17])
        self.stage7 = nn.Sequential(list(features.children())[17])
        if self.classifier:
            self.last_conv = nn.Sequential(list(features.children())[18])
            self.fc = mobilenet_v2(self.pretrained).classifier
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
        for i in range(1, 8):
            stage = getattr(self, 'stage{}'.format(i))
            x = stage(x)
            if i in self.out_stages and not self.classifier:
                output.append(x)
        if self.classifier:
            x = self.last_conv(x)
            x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
            x = self.fc(x)
            return x
        return tuple(output)


    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def load_pretrained_weights(self):
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))


if __name__=="__main__":
    model =MobileNetV2('mobilenet_v2', classifier=True, pretrained=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    for o in out:
        print(o.shape)