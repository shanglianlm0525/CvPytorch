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
import shutil


def showImgAndMask():
    imgpath = 'E:\\data\\ADE20K_2016_07_26\\images2\\training\\a\\abbey\\ADE_train_00000970.jpg'
    segpath = 'E:\\data\\ADE20K_2016_07_26\\images2\\training\\a\\abbey\\ADE_train_00000970_seg.png'
    img = cv2.imread(imgpath)
    seg = cv2.imread(segpath)

    cv2.imshow('img',img)
    cv2.imshow('seg',seg)
    cv2.waitKey(0)


def transImgAndMask():
    path = 'E:\\data\\ADE20K_2016_07_26\\images2\\validation'
    imglist = glob(path+"\\*\\*\\*.jpg")
    masklist = glob(path+"\\*\\*\\*seg.png")

    imgnew = 'E:\\data\\ADE20K_2016_07_26\\images\\validation'
    masknew = 'E:\\data\\ADE20K_2016_07_26\\annotations\\validation'
    for imgpath,maskpath in zip(imglist,masklist):
        imgbasename = os.path.basename(imgpath)
        maskbasename = os.path.basename(maskpath)
        shutil.copyfile(imgpath,os.path.join(imgnew,imgbasename))
        shutil.copyfile(maskpath,os.path.join(masknew,maskbasename))


def produceDict():
    seg_name = ['wall' 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ',
        'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth',
        'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car',
        'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug',
        'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe',
        'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
        'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
        'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path',
        'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door',
        'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table',
        'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove',
        'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
        'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
        'chandelier', 'awning', 'streetlight', 'booth', 'television receiver',
        'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister',
        'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van',
        'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything',
        'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent',
        'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank',
        'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake',
        'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce',
        'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen',
        'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
        'clock', 'flag']

    for seg in seg_name:
        print(' - '+seg+': 1')

produceDict()
print('finished!')
