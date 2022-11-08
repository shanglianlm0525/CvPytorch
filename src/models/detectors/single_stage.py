# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/26 10:41
# @Author : liumin
# @File : single_stage.py

import torch
import torch.nn as nn
from torch import Tensor

from src.models.backbones import build_backbone
from src.models.heads import build_head
from src.models.necks import build_neck
from src.losses import build_loss

from src.utils.config import CommonConfiguration
from src.models.detectors.base_detector import BaseDetector
from src.utils.misc import add_prefix


class SingleStageDetector(BaseDetector):
    def __init__(self, dictionary=None, model_cfg=None):
        super(SingleStageDetector, self).__init__()
        self.dictionary = dictionary
        self.model_cfg = model_cfg
        self.dummy_input = torch.zeros(1, 3, 640, 640)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        self.setup_extra_params()
        # BACKBONE
        self.backbone = build_backbone(self.model_cfg.BACKBONE)
        # NECK
        if self.model_cfg.NECK is not None:
            self.neck = build_neck(self.model_cfg.NECK)
        # HEAD
        self.head = build_head(self.model_cfg.HEAD)
        # AUX_HEAD
        if self.model_cfg.AUX_HEAD is not None:
            assert self.model_cfg.AUX_LOSS is not None, f'{self.model_cfg.AUX_LOSS} must not be None'
            assert type(self.model_cfg.AUX_HEAD) == type(self.model_cfg.AUX_LOSS)
            assert (isinstance(self.model_cfg.AUX_HEAD, (list, tuple)) and len(self.model_cfg.AUX_HEAD) == len(
                self.model_cfg.AUX_LOSS)) \
                   or (not isinstance(self.model_cfg.AUX_HEAD, (list, tuple)))

            self.auxiliary_head = nn.ModuleList()
            if isinstance(self.model_cfg.AUX_HEAD, (dict, CommonConfiguration)):
                self.auxiliary_head.append(build_head(self.model_cfg.AUX_HEAD))
            elif isinstance(self.model_cfg.AUX_HEAD, (list, tuple)):
                for head_cfg in self.model_cfg.AUX_HEAD:
                    self.auxiliary_head.append(build_head(head_cfg))
            else:
                raise TypeError(f'self.model_cfg.AUX_HEAD must be a dict or sequence of dict,\
                                               but got {type(self.model_cfg.AUX_HEAD)}')
        # LOSS
        self.loss = nn.ModuleList()
        if isinstance(self.model_cfg.LOSS, (dict, CommonConfiguration)):
            self.loss.append(build_loss(self.model_cfg.LOSS))
        elif isinstance(self.model_cfg.LOSS, (list, tuple)):
            for loss_cfg in self.model_cfg.LOSS:
                self.loss.append(build_loss(loss_cfg))
        else:
            raise TypeError(f'self.model_cfg.LOSS must be a dict or sequence of dict,\
                               but got {type(self.model_cfg.LOSS)}')

        # AUX_LOSS
        if self.model_cfg.AUX_LOSS is not None:
            self.auxiliary_loss = nn.ModuleList()
            if isinstance(self.model_cfg.AUX_LOSS, (dict, CommonConfiguration)):
                self.auxiliary_loss.append(build_loss(self.model_cfg.AUX_LOSS))
            elif isinstance(self.model_cfg.AUX_LOSS, (list, tuple)):
                for loss_cfg in self.model_cfg.AUX_LOSS:
                    self.auxiliary_loss.append(build_loss(loss_cfg))
            else:
                raise TypeError(f'self.model_cfg.AUX_LOSS must be a dict or sequence of dict,\
                                               but got {type(self.model_cfg.AUX_LOSS)}')


    def setup_extra_params(self):
        pass


    def trans_specific_format(self, imgs, targets):
        pass


    def loss_forward(self, preds, targets, loss):
        assert isinstance(preds, Tensor) and preds.ndim == 4, \
            f'preds must be Tensor type, and ndim == 4, but got {type(preds)} and {preds.ndim}'
        assert isinstance(targets, Tensor) and targets.ndim == 3, \
            f'targets must be Tensor type, and ndim == 3, but got {type(targets)} and {targets.ndim}'
        losses = dict()

        if not isinstance(loss, (list, nn.ModuleList)):
            loss = [loss]
        for l in loss:
            if l.loss_name not in losses:
                losses[l.loss_name] = l(preds, targets)
            else:
                losses[l.loss_name] += l(preds, targets)
        return losses

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        """
        Args:
            imgs: (Tensor)
                Shape: N x C x H x W
            targets: (Tensor | list[Tensor])
                Shape: N x H x W.
            mode: (str) run type of forward.
                Options: 'train', 'val', 'infer'.
                Default: 'infer'.
        """
        imgs, targets = self.trans_specific_format(imgs, targets)

        feats = self.backbone(imgs)
        feats = self.neck(feats)
        if isinstance(feats, tuple):
            feats, aux_feats = feats
        preds = self.head(feats)

        if mode == 'infer':

            return torch.argmax(feats, dim=1)
        else:
            losses = self.loss_forward(preds, targets["gt"],  self.loss)


            if mode == 'val':

                return
            else:
                losses = self.loss_forward(preds, targets, self.loss)

                if self.with_auxiliary_head:
                    if aux_feats is not None:
                        for i, (aux_head, aux_feat, auxiliary_loss) in enumerate(zip(self.auxiliary_head, aux_feats, self.auxiliary_loss)):
                            aux_pred = aux_head(aux_feat)
                            aux_losses = self.loss_forward(aux_pred, targets, auxiliary_loss)
                            losses.update(add_prefix(aux_losses, 'aux' + str(i)))
                    else:
                        for i, (aux_head, auxiliary_loss) in enumerate(zip(self.auxiliary_head, self.auxiliary_loss)):
                            aux_pred = aux_head(feats)
                            aux_losses = self.loss_forward(aux_pred, targets, auxiliary_loss)
                            losses.update(add_prefix(aux_losses, 'aux' + str(i)))

                losses['loss'] = sum(losses.values())
                return losses