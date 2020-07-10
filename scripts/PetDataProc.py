# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/11 14:38
# @Author : liumin
# @File : PetDataProc.py
import os
import shutil
import sys
import cv2
from glob import glob
import scipy.io as scio


def arrangeAllImg():
    rootpath = 'C:\\Users\\liumin\\Desktop\\Pet'
    paths = glob(os.path.join(rootpath+'\\images', '*.jpg'))
    dogpath = os.path.join(rootpath,'all\\dog')
    if not os.path.exists(dogpath):
        os.makedirs(dogpath)
    catpath = os.path.join(rootpath,'all\\cat')
    if not os.path.exists(catpath):
        os.makedirs(catpath)

    for idx,imgPath in enumerate(paths):
        [fpath, fname] = os.path.split(imgPath)
        if fname.islower():
            shutil.copy(imgPath, os.path.join(dogpath, fname))
        else:
            shutil.copy(imgPath, os.path.join(catpath, fname))


## split sample 7:2:1

def splitSample():
    rootpath = 'C:\\Users\\liumin\\Desktop\\Pet'
    for label in ['dog','cat']:
        imgpath = os.path.join(rootpath,'all\\'+label)
        imgpath_train = os.path.join(rootpath,'train\\'+label)
        imgpath_val = os.path.join(rootpath,'val\\'+label)
        imgpath_test = os.path.join(rootpath,'test\\'+label)

        if not os.path.exists(imgpath_train):
            os.makedirs(imgpath_train)
        if not os.path.exists(imgpath_val):
            os.makedirs(imgpath_val)
        if not os.path.exists(imgpath_test):
            os.makedirs(imgpath_test)

        paths = glob(os.path.join(imgpath, '*.jpg'))
        for idx,imgPath in enumerate(paths):
            [fpath, fname] = os.path.split(imgPath)
            if idx <= 0.7*len(paths):
                shutil.copy(imgPath, os.path.join(imgpath_train,fname))
            elif 0.7*len(paths) < idx <=0.9*len(paths):
                shutil.copy(imgPath, os.path.join(imgpath_val,fname))
            else:
                shutil.copy(imgPath, os.path.join(imgpath_test, fname))

splitSample()
print('finished!')