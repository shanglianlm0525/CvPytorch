# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/16 22:45
# @Author : liumin
# @File : compare_cityspaces_trainids_and_labelids.py

import copy
from PIL import Image
import cv2
import numpy as np

invalid_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_map = dict(zip(valid_classes, range(19)))

labelIds_path = 'aachen_000000_000019_gtFine_labelIds.png'
labelTrainIds_path = 'aachen_000000_000019_gtFine_labelTrainIds.png'
labelIds = cv2.imread(labelIds_path, 0)
labelTrainIds = cv2.imread(labelTrainIds_path, 0)

print(labelIds)
print(labelTrainIds)

labelIds_new  = copy.deepcopy(labelIds)
# Put all void classes to zero
for _voidc in invalid_classes:
    labelIds_new[labelIds_new == _voidc] = 255

# index from zero 0:18
for _validc in valid_classes:
    labelIds_new[labelIds_new == _validc] = class_map[_validc]

print(labelIds_new)
aaa = labelIds_new == labelTrainIds
bbb = labelIds_new != labelTrainIds
print(np.sum(np.sum(aaa)), np.sum(np.sum(aaa))/(1024*2048))
print(np.sum(np.sum(bbb)), np.sum(np.sum(bbb))/(1024*2048))

print(labelIds[bbb], np.unique(labelIds[bbb]))
print(labelIds_new[bbb], np.unique(labelIds_new[bbb]))
print(labelTrainIds[bbb], np.unique(labelTrainIds[bbb]))

label1 = np.zeros_like(labelIds_new)
label2 = np.zeros_like(labelIds_new)
label3 = np.zeros_like(labelIds_new)

label1[labelIds_new==10] = 255
label2[labelTrainIds==10] = 255

print(np.sum(np.sum(label1))/255)
print(np.sum(np.sum(label2))/255)

label3[label1!=label2] = 255
    

label1 = cv2.resize(label1, (1200, 800))
label2 = cv2.resize(label2, (1200, 800))
label3 = cv2.resize(label3, (1200, 800))
cv2.imshow('label1', label1)
cv2.imshow('label2', label2)
cv2.imshow('label3', label3)
'''
cv2.imshow('labelIds', labelIds)
cv2.imshow('labelTrainIds', labelTrainIds)
cv2.imshow('labelIds_new', labelIds_new)
'''
cv2.waitKey(0)