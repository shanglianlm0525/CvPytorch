# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 10:58
# @Author : liumin
# @File : misc.py

def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}_{name}'] = value

    return outputs