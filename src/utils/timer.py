# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/11 17:11
# @Author : liumin
# @File : timer.py

import time
from functools import wraps

def runtime_info(f):
    @wraps(f)
    def info(*args, **kwargs):
        print('begin to execution function : %s ' % (f.__qualname__,))
        start = time.time()
        rst = f(*args, **kwargs)
        print('%s function execution time is : %f s' % (f.__qualname__,(time.time() - start)))
        return rst
    return info


class Timer(object):
    """
        A simple timer (adapted from Detectron).
    """
    def __init__(self):
        self.total_time = None
        self.calls = None
        self.start_time = None
        self.diff = None
        self.average_time = None
        self.reset()

    def tic(self):
        # using time.time as time.clock does not normalize for multithreading
        self.start_time = time.time()

    def toc(self):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0
        self.average_time = 0.0