# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/24 18:19
# @Author : liumin
# @File : ssd_old.py

import torch
import torch.nn as nn
import torch.nn.functional as torchF

from src.models.ext.ssd.detection import Detect
from src.models.ext.ssd.l2norm import L2Norm

from src.models.ext.ssd.config import voc, coco
from src.models.ext.ssd.multibox_loss import MultiBoxLoss
from src.models.ext.ssd.prior_box import PriorBox

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [4, 6, 6, 6, 4, 4],
}

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


class SSD(nn.Module):
    '''
        SSD: Single Shot MultiBox Detector
        https://arxiv.org/pdf/1512.02325.pdf
    '''
    def __init__(self, dictionary=None):
        super(SSD, self).__init__()
        self.dictionary = dictionary
        self.input_size = [300,300]

        self.dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])

        self._num_classes = len(self.dictionary)+1
        self._category = [v for d in self.dictionary for v in d.keys()]
        self._weight = [d[v] for d in self.dictionary for v in d.keys() if v in self._category]

        self._priorbox = PriorBox(coco)
        self._priors = self._priorbox.forward()

        base_, extras_, head_ = multibox(vgg(base[str(self.input_size[0])], 3),
                                         add_extras(extras[str(self.input_size[0])], 1024),
                                         mbox[str(self.input_size[0])], self._num_classes)

        # SSD network
        self.vgg = nn.ModuleList(base_)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras_)

        self.loc = nn.ModuleList(head_[0])
        self.conf = nn.ModuleList(head_[1])

        self._criterion = MultiBoxLoss(self._num_classes, 0.5, True, 0, True, 3, 0.5, False)

        self.softmax = nn.Softmax(dim=-1)
        self._detect = Detect(self._num_classes, 0, 200, 0.01, 0.45)

        self.init_params()

        self.vgg.load_state_dict(torch.load('src/models/weights/vgg16_reducedfc.pth'))

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, imgs, labels=None, mode='infer', **kwargs):
        """Applies network layers and ops on input image(s) x.

                Args:
                    x: input image or batch of images. Shape: [batch,3,300,300].

                Return:
                    Depending on phase:
                    test:
                        Variable(tensor) of output class label predictions,
                        confidence score, and corresponding location predictions for
                        each object detected. Shape: [batch,topk,7]

                    train:
                        list of concat outputs from:
                            1: confidence layers, Shape: [batch*num_priors,num_classes]
                            2: localization layers, Shape: [batch,num_priors*4]
                            3: priorbox layers, Shape: [2,num_priors*4]
                """
        sources = list()
        loc = list()
        conf = list()

        x = imgs
        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = torchF.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        device_id = labels[0].data.device if isinstance(labels,list) else labels.data.device

        if mode == 'infer':
            out = self._detect(
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(conf.size(0), -1, self._num_classes)),  # conf preds
                self._priors.type(type(x.data))  # default boxes
            )
            return out
        else:
            losses = {}

            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self._num_classes),
                self._priors.cuda(device=device_id)
            )

            losses['loc_loss'], losses['conf_loss'] = self._criterion(output, labels)
            losses['all_loss'] = losses['loc_loss'] + losses['conf_loss']

            if mode == 'val':
                performances = {}
                performances['all_perf'] = losses['all_loss']
                '''
                out = self._detect(
                    loc.view(loc.size(0), -1, 4),  # loc preds
                    self.softmax(conf.view(conf.size(0), -1, self._num_classes)),  # conf preds
                    self._priors.cuda(device=device_id)  # default boxes
                )
                detections = out.data
                print('detections.shape',detections.shape)
                print(detections)

                threshold = 0.6
                for i in range(detections.size(1)):
                    for j in range(detections.size(2)):
                        score = detections[0, i, j, 0]
                        if score >= threshold:
                            pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                            # label_name = labelmap[i - 1]
                '''

                return losses, performances
            else:
                return losses