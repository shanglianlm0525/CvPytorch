# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/30 13:16
# @Author : liumin
# @File : VOCProc.py

import os
import shutil
import sys
import cv2
from glob import glob


def produceImgAndLabelList():
    root_path = '/home/lmin/data/VOCdevkit/'

    stages = ['train','val']
    for stage in stages:
        det_txt = open(root_path+stage+'.txt','a')

        for year in ['VOC2007', 'VOC2012']:
            txtpath = os.path.join(root_path,year,'ImageSets/Main/'+stage+'.txt')
            imgpath = os.path.join(root_path,year,'JPEGImages','%s.jpg')
            xmlpath = os.path.join(root_path,year,'Annotations','%s.xml')
            for line in open(txtpath):
                imgpath1 = (imgpath % line.strip()).replace(root_path, '')
                print(imgpath1, (xmlpath % line.strip()).replace(root_path,''))
                if os.path.isfile(os.path.join(root_path, imgpath1)):
                    det_txt.write(imgpath1+' '+(xmlpath % line.strip()).replace(root_path,'')+'\n')

        det_txt.close()

produceImgAndLabelList()
print('finished!')
