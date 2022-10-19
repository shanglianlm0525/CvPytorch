# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 15:07
# @Author : liumin
# @File : segnext_new.py

from src.models.segmentors.encoder_decoder import EncoderDecoder


class SegNeXt(EncoderDecoder):
    def __init__(self, *args, **kwargs):
        super(SegNeXt, self).__init__(*args, **kwargs)