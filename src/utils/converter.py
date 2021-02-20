# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/20 11:12
# @Author : liumin
# @File : converter.py


def to_tuple(size):
    if isinstance(size, (list, tuple)):
        return tuple(size)
    elif isinstance(size, (int, float)):
        return tuple((size, size))
    else:
        raise ValueError('Unsupport data type: {}'.format(type(size)))
