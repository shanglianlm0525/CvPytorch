# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/4 9:59
# @Author : liumin
# @File : CityspacesProc.py

import os
import sys
import numpy as np
from glob2 import glob


def produceImgAndLabelsList():
    root_path = '/home/lmin/data/cityscapes/cityscapes/'

    stages = ['test'] # ['train', 'val','test']
    for stage in stages:
        seg_txt = open(root_path + stage + '.txt', 'a')

        images_txt = open(root_path+stage +'Images.txt', 'r')
        labels_txt = open(root_path+stage +'Labels.txt', 'r')

        for line1,line2 in zip(images_txt,labels_txt):
            print(line1.strip()+' '+line2.strip())
            line2 = line2.replace('_labelTrainIds.png','_labelIds.png')
            seg_txt.write(line1.strip()+' '+line2.strip() + '\n')
        images_txt.close()
        labels_txt.close()
        seg_txt.close()

produceImgAndLabelsList()
print('finished!')