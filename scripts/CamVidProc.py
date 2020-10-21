# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/10/14 9:58
# @Author : liumin
# @File : CamVidProc.py

import os
import shutil
import sys
import cv2
from glob import glob
import numpy as np

def produceImgAndLabel():
    root_path = '/home/lmin/data/CamVid/'

    stages = ['train', 'val', 'test']
    for stage in stages:
        seg_txt = open(root_path + stage + '.txt', 'a')

        imgpath = glob(os.path.join(root_path, stage,'images/*.png'))
        txtpath = glob(os.path.join(root_path, stage,'masks/*.png'))

        for imgline,txtline in zip(imgpath,txtpath):
            print(imgline.replace(root_path, ''))
            print(txtline.replace(root_path, ''))

            seg_txt.write(imgline.replace(root_path, '') + ' ' + txtline.replace(root_path, '') + '\n')

        seg_txt.close()

produceImgAndLabel()

def produceImgList():
    root_path = '/home/lmin/data/portrait/'

    stages = ['train', 'val']
    for stage in stages:
        seg_txt = open(root_path + stage + '_2.txt', 'a')

        imgpath = glob(os.path.join(root_path, stage,'images/*.png'))

        for imgline in imgpath:
            print(imgline.replace(root_path, ''))
            seg_txt.write(imgline.replace(root_path, '') + '\n')

        seg_txt.close()

# produceImgList()