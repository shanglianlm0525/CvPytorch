# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/8 13:55
# @Author : liumin
# @File : torch_utils.py

import os
import numpy as np
import random
import torch


def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    if seed == 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_class_name(full_class_name):
    return full_class_name.split(".")[-1]