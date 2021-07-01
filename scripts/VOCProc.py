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
        det_txt = open(root_path+stage+'_det.txt','a')
        seg_txt = open(root_path+stage+'_seg.txt','a')

        for year in ['VOC2007', 'VOC2012']:
            txtpath = os.path.join(root_path,year,'ImageSets/Main/'+stage+'.txt')
            imgpath = os.path.join(root_path,year,'JPEGImages','%s.jpg')
            xmlpath = os.path.join(root_path,year,'Annotations','%s.xml')
            for line in open(txtpath):
                imgpath1 = (imgpath % line.strip()).replace(root_path, '')
                print(imgpath1, (xmlpath % line.strip()).replace(root_path,''))
                if os.path.isfile(os.path.join(root_path, imgpath1)):
                    det_txt.write(imgpath1+' '+(xmlpath % line.strip()).replace(root_path,'')+'\n')

        for year in ['VOC2012']:
            txtpath = os.path.join(root_path,year,'ImageSets/Segmentation/'+stage+'.txt')
            imgpath = os.path.join(root_path,year,'JPEGImages','%s.jpg')
            pngpath = os.path.join(root_path,year,'SegmentationClass','%s.png')
            for line in open(txtpath):
                imgpath1 = (imgpath % line.strip()).replace(root_path, '')
                pngpath1 = (pngpath % line.strip()).replace(root_path, '')
                print(imgpath1, pngpath1)
                if os.path.isfile(os.path.join(root_path, imgpath1)) and os.path.isfile(os.path.join(root_path, pngpath1)):
                    seg_txt.write(imgpath1 + ' ' + pngpath1 + '\n')


            train_txtpath = os.path.join(root_path, year, 'train_aug.txt')
            with open(train_txtpath, "r") as f:
                train_list = f.readlines()
            train_list = [t.strip() for t in train_list]

            imgpath = os.path.join(root_path, year, 'JPEGImages', '%s.jpg')
            aug_pngpath = os.path.join(root_path, year, 'SegmentationClassAug')
            for pngname in os.listdir(aug_pngpath):
                pname = pngname.split('.')[0]
                if (pname in train_list and stage=='train') or (pname not in train_list and stage=='val'):
                    imgpath1 = (imgpath % pname.strip()).replace(root_path, '')
                    pngpath1 = os.path.join(aug_pngpath, pngname).replace(root_path, '')
                    print(imgpath1, pngpath1)
                    if os.path.isfile(os.path.join(root_path, imgpath1)):
                        seg_txt.write(imgpath1 + ' ' + pngpath1 + '\n')

                '''
                aug_pngpath = os.path.join(root_path, year, 'SegmentationClassAug', '%s.png')
                for line in open(txtpath):
                    imgpath1 = (imgpath % line.strip()).replace(root_path, '')
                    aug_pngpath1 = (aug_pngpath % line.strip()).replace(root_path, '')
                    print(imgpath1, aug_pngpath1)
                    if os.path.isfile(os.path.join(root_path, imgpath1)) and os.path.isfile(
                            os.path.join(root_path, aug_pngpath1)):
                        seg_txt.write(imgpath1 + ' ' + aug_pngpath1 + '\n')


                aug_pngpath = os.path.join(root_path, year, 'SegmentationClassAug')
                for pngname in os.listdir(aug_pngpath):
                    pname = pngname.split('.')[0]
                    imgpath1 = (imgpath % pname.strip()).replace(root_path, '')
                    pngpath1 = os.path.join(aug_pngpath, pngname).replace(root_path, '')
                    print(imgpath1, pngpath1)
                    if os.path.isfile(os.path.join(root_path, imgpath1)):
                        seg_txt.write(imgpath1+' '+pngpath1+'\n')
                '''
        det_txt.close()
        seg_txt.close()

produceImgAndLabelList()
print('finished!')
