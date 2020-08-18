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


root_path = '/home/lmin/data/VOCdevkit/20200730/'

stages = ['train','val']
years = ['VOC2007','VOC2012']

for stage in stages:
    det_txt = open(root_path+stage+'_det.txt','a')
    seg_txt = open(root_path+stage+'_seg.txt','a')

    for year in years:
        txtpath = os.path.join(root_path,year,'ImageSets/Main/'+stage+'.txt')
        imgpath = os.path.join(root_path,year,'JPEGImages','%s.jpg')
        xmlpath = os.path.join(root_path,year,'Annotations','%s.xml')
        pngpath = os.path.join(root_path,year,'SegmentationClass','%s.png')
        for line in open(txtpath):
            print((imgpath % line.strip()).replace(root_path,''))
            print((xmlpath % line.strip()).replace(root_path,''))
            print((pngpath % line.strip()).replace(root_path,''))
            det_txt.write((imgpath % line.strip()).replace(root_path,'')+' '+(xmlpath % line.strip()).replace(root_path,'')+'\n')
            seg_txt.write((imgpath % line.strip()).replace(root_path,'')+' '+(pngpath % line.strip()).replace(root_path,'')+'\n')

    det_txt.close()
    seg_txt.close()

print('finished!')
