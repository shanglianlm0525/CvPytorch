# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/7/28 9:16
# @Author : liumin
# @File : COCOProc.py
import json
import os
import shutil
import sys
import cv2
from glob import glob
import shutil


def produceDict():
    seg_name = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire', 'hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard',
                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                'keyboard', 'cell phone', 'microwave oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear', 'hair drier', 'toothbrush']

    for seg in seg_name:
        print(' - '+seg+': 1')



def procJson():
    data = None
    with open('instances_val2017.json', 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        if len(strF) > 0:
            data = json.loads(strF)

    info = data['info']
    licenses = data['licenses']
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']



    valid = ['000000000139.jpg','000000000285.jpg']

    # valid = ['000000000025.jpg', '000000000030.jpg']

    images_valid = []
    annotations_valid = []
    for ig in images:
        if ig['file_name'] in valid:
            print(ig)
            images_valid.append(ig)
            for an in annotations:
                if ig['id'] == an['image_id']:
                    print(an)
                    annotations_valid.append(an)

    data['images'] = images_valid
    data['annotations'] = annotations_valid

    with open("instances_test_val2017.json", "w") as f:
        json.dump(data, f)


# procJson()

def validJson():
    data = None
    with open('/home/lmin/data/coco/annotations/instances_val2017.json', 'r', encoding='utf-8') as load_f:
        strF = load_f.read()
        if len(strF) > 0:
            data = json.loads(strF)

    info = data['info']
    licenses = data['licenses']
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']


    validpath = glob(os.path.join('/home/lmin/data/coco/images/val2017','*.jpg'))
    valid = [os.path.basename(path) for path in validpath]

    images_valid = []
    annotations_valid = []
    for ig in images:
        if ig['file_name'] in valid:
            # print(ig)
            images_valid.append(ig)
            for an in annotations:
                if ig['id'] == an['image_id']:
                    # print(an)
                    annotations_valid.append(an)

    data['images'] = images_valid
    data['annotations'] = annotations_valid

    with open("/home/lmin/data/coco/annotations2/instances_val2017.json", "w") as f:
        json.dump(data, f)


# validJson()

print('finished!')