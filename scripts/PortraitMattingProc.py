# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/8 14:40
# @Author : liumin
# @File : PortraitMattingProc.py


import os
import shutil
import sys
import cv2
from glob import glob

import torch
import torchvision
from PIL import Image # 导入PIL库
import numpy as np


def separateImgAndLabel():
    rootpath = 'C:\\Users\\liumin\\Desktop\\dataset\\testing'
    paths = glob(os.path.join(rootpath, '*.png'))

    for idx,imgPath in enumerate(paths):
        [fpath, fname] = os.path.split(imgPath)
        print(imgPath)
        if '_matte' in fname:
            shutil.copy(imgPath, os.path.join('C:\\Users\\liumin\\Desktop\\XX', fname))
        else:
            shutil.copy(imgPath, os.path.join('C:\\Users\\liumin\\Desktop\\YY', fname))

separateImgAndLabel()

def procLabelToStandardFormat():
    rootpath = 'C:\\Users\\liumin\\Desktop\\XXX'
    paths = glob(os.path.join(rootpath, '*.png'))

    newpath = 'C:\\Users\\liumin\\Desktop\\ZZZ'
    for idx, imgPath in enumerate(paths):
        [fpath, fname] = os.path.split(imgPath)
        print(imgPath)
        img = cv2.imread(imgPath,0)
        mask0 = np.zeros((img.shape[0],img.shape[1]))
        mask1 = np.zeros((img.shape[0],img.shape[1]))
        mask0[img<1] = 1
        mask1[img>1] = 1

        stackImg = np.zeros((img.shape[0],img.shape[1],1))
        print(stackImg.shape)

        cv2.imwrite("stackImg.png", stackImg)
        cv2.imshow('mask0', mask0)
        cv2.imshow('mask1', mask1)
        cv2.imshow('img',img)
        cv2.waitKey(0)

# procLabelToStandardFormat()