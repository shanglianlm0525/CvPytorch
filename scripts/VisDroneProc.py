# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/11/17 16:59
# @Author : liumin
# @File : VisDroneProc.py

import os
import shutil
import sys
import cv2
from glob import glob


def produceImgAndLabel():
    root_path = '/home/lmin/data/visdrone/'

    stages = ['train', 'val', 'test']
    for stage in stages:
        seg_txt = open(root_path + stage + '.txt', 'a')

        imgpath = glob(os.path.join(root_path, stage,'images/*.jpg'))
        txtpath = glob(os.path.join(root_path, stage,'annotations/*.txt'))

        for imgline,txtline in zip(imgpath,txtpath):
            print(imgline.replace(root_path, ''))
            print(txtline.replace(root_path, ''))

            seg_txt.write(imgline.replace(root_path, '') + ' ' + txtline.replace(root_path, '') + '\n')

        seg_txt.close()

produceImgAndLabel()

