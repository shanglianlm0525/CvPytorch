# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/24 16:18
# @Author : liumin
# @File : MiniImageNetProc.py

import os
import json

# import pandas as pd
import random

from PIL import Image
import matplotlib.pyplot as plt


def read_csv_classes(csv_dir: str, csv_name: str):
    data = []
    label_set = []
    with open(os.path.join(csv_dir, csv_name)) as file:
        for line in file:
            a, b = line.strip('\n').split(',')
            data.append(a)
            label_set.append(b)
    return dict([(d, l) for d, l in zip(data[1:], label_set[1:])])


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def calculate_split_info(path: str, label_dict: dict, rate: float = 0.2):
    # read all images
    image_dir = os.path.join(path, "images")
    images_list = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    print("find {} images in dataset.".format(len(images_list)))

    train_data = read_csv_classes(path, "train.csv")
    val_data = read_csv_classes(path, "val.csv")
    test_data = read_csv_classes(path, "test.csv")


    # concat csv data
    data = dict(train_data, **val_data, **test_data)
    print("total data shape: {}".format(len(data)))

    tmp = list(set(list(data.values())))
    classes_label = dict([(label, index) for index, label in enumerate(tmp)])
    #   - ants: 1
    for t in tmp:
        print('  - ' + label_dict[t] + ': 1')

    data = random_dic(data)

    train_txt = open(os.path.join(path, 'train.txt'), 'a')
    val_txt = open(os.path.join(path, 'val.txt'), 'a')

    data_rate = len(data) * rate
    for i, (k, v) in enumerate(data.items()):
        # print('%s:%s' % (k, v))
        if i < data_rate:
            val_txt.write(os.path.join('image', k) + ' ' + str(classes_label[v]) + '\n')
        else:
            train_txt.write(os.path.join('image', k) + ' ' + str(classes_label[v]) + '\n')

    train_txt.close()
    val_txt.close()

    print('finish!')


def generateSampleAndDict():
    data_dir = "/home/lmin/data/mini-imagenet/"  # 指向数据集的根目录
    json_path = os.path.join(data_dir, "imagenet_class_index.json")  # 指向imagenet的索引标签文件

    # load imagenet labels
    label_dict = json.load(open(json_path, "r"))
    label_dict = dict([(v[0], v[1]) for k, v in label_dict.items()])

    calculate_split_info(data_dir, label_dict)


if __name__ == '__main__':
    generateSampleAndDict()
