# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/10/4 10:58
# @Author : liumin
# @File : misc.py
import math


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


def make_divisible(x: float,
                   widen_factor: float = 1.0,
                   divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x
