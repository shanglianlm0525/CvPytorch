# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/7 10:16
# @Author : liumin
# @File : ADE20kProc.py

import os
import shutil
import sys
import cv2
from glob import glob
import scipy.io as scio


imgpath = 'C:\\Users\\liumin\\Desktop\\code (1)\\ADE_train_00000970.jpg'
segpath = 'C:\\Users\\liumin\\Desktop\\code (1)\\ADE_train_00000970_seg.png'
img = cv2.imread(imgpath)
seg = cv2.imread(segpath)

cv2.imshow('img',img)
cv2.imshow('seg',seg)
cv2.waitKey(0)