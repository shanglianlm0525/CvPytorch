# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/30 18:40
# @Author : liumin
# @File : vision_transformer.py

import math
import torch
import torch.nn as nn
from torch.utils import model_zoo
import torchvision.models as models
from torchvision.models.vision_transformer import model_urls

'''
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://arxiv.org/pdf/2010.11929.pdf
'''


class VisionTransformer(nn.Module):
    def __init__(self, subtype='vit_b_16', out_stages=[2, 3, 4], output_stride=16, classifier=False, num_classes=1000,
                     pretrained=True, backbone_path=None):

        super(VisionTransformer, self).__init__()
        self.subtype = subtype
        self.out_stages = out_stages
        self.output_stride = output_stride  # 8, 16, 32
        self.classifier = classifier
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone_path = backbone_path

        self.image_size = 224

        if self.subtype == 'vit_b_16':
            vit = models.vit_b_16(pretrained=self.pretrained)
        elif self.subtype == 'vit_b_32':
            vit = models.vit_b_32(pretrained=self.pretrained)
        elif self.subtype == 'vit_l_16':
            vit = models.vit_l_16(pretrained=self.pretrained)
        elif self.subtype == 'vit_l_32':
            vit = models.vit_l_32(pretrained=self.pretrained)
        else:
            raise NotImplementedError

        self.vit = vit
        self.conv_proj = vit.conv_proj  # x2
        self.encoder = vit.encoder
        if self.classifier:
            self.heads = vit.heads
            self.heads.head = nn.Linear(self.heads.head.in_features, self.num_classes)
            self.out_channels = self.num_classes

        if self.pretrained:
            self.load_pretrained_weights()
        else:
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.weight.shape[1])
                nn.init.uniform_(m.weight, -init_range, init_range)

    def load_pretrained_weights(self):
        url = model_urls[self.subtype]
        if url is not None:
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif self.backbone_path is not None:
            print('=> loading pretrained model {}'.format(self.backbone_path))
            self.load_state_dict(torch.load(self.backbone_path))

    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.vit.patch_size
        torch._assert(h == self.image_size, "Wrong image height!")
        torch._assert(w == self.image_size, "Wrong image width!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.vit.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x



if __name__=='__main__':
    model = VisionTransformer('vit_b_16', classifier=True)
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)