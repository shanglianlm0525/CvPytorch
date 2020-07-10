# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/11 10:11
# @Author : liumin
# @File : FlowerDataProc.py

import os
import shutil
import sys
import cv2
from glob import glob
import scipy.io as scio


def arrangeAllImg17():
    rootpath = 'C:\\Users\\liumin\\Desktop\\Flower102'

    f = open(os.path.join(rootpath, 'jpg\\files.txt'),'r')
    lines = f.readlines()
    for idx, line in enumerate(lines):
        id = idx // 80 + 1
        line = line.strip()
        print(idx+1,id, line)

        newpath = os.path.join(rootpath+'\\all',str(id))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        shutil.copyfile(os.path.join(rootpath+'\\jpg',line),os.path.join(newpath,line))

# arrangeAllImg()

def arrangeAllImg102():
    rootpath = 'C:\\Users\\liumin\\Desktop\\Flower102'
    data = scio.loadmat(os.path.join(rootpath, 'imagelabels.mat'))
    print(data['labels'])
    paths = glob(os.path.join(rootpath, 'jpg\\*.jpg'))
    for imgPath, label in zip(paths, data['labels'].T.tolist()):
        print(imgPath,label[0])
        [fpath, fname] = os.path.split(imgPath)
        newpath = os.path.join(rootpath + '\\all', str(label[0]))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        shutil.copyfile(imgPath, os.path.join(newpath, fname))

# arrangeAllImg102()

## split sample 7:2:1

def splitSample():
    rootpath = 'C:\\Users\\liumin\\Desktop\\Flower102'
    for r in range(102):
        label = str(r+1)

        imgpath = rootpath+'\\all\\'+label
        imgpath_train = rootpath+'\\train\\'+label
        imgpath_val = rootpath+'\\val\\'+label
        imgpath_test = rootpath+'\\test\\'+label

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

