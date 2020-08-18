# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/8/17 14:54
# @Author : liumin
# @File : PennFudanProc.py

import os
import sys
import re

import cv2
import numpy as np
from glob2 import glob
from lxml.etree import Element, SubElement, tostring
import pprint
from xml.dom.minidom import parseString


def produceImgAndLabel():
    root_path = '/home/lmin/data/PennFudanPed/'

    imgpath = sorted(glob(os.path.join(root_path, 'PNGImages/*.png')))
    txtpath = sorted(glob(os.path.join(root_path, 'PedMasks/*.png')))

    train_seg_txt = open(root_path + 'train' + '_ins.txt', 'a')
    val_seg_txt = open(root_path + 'val' + '_ins.txt', 'a')


    for i,(imgline,txtline) in enumerate(zip(imgpath,txtpath)):
        print(imgline.replace(root_path, ''))
        print(txtline.replace(root_path, ''))
        if i%5 != 0:
            train_seg_txt.write(imgline.replace(root_path, '') + ' ' + txtline.replace(root_path, '') + '\n')
        else:
            val_seg_txt.write(imgline.replace(root_path, '') + ' ' + txtline.replace(root_path, '') + '\n')

    train_seg_txt.close()
    val_seg_txt.close()

# produceImgAndLabel()


def produceXMl(imgname,height,width,busbars_coords):
    node_root = Element('annotation')

    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'PVD'

    node_filename = SubElement(node_root, 'filename')
    node_filename.text = imgname

    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(width)

    node_height = SubElement(node_size, 'height')
    node_height.text = str(height)

    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'

    for busbars in busbars_coords:
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = 'pedestrian'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(busbars[0])
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(busbars[1])
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(busbars[2])
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(busbars[3])

    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)
    # print(xml)
    return xml

def txt2voc():
    root_path = '/home/lmin/data/PennFudanPed'
    txtlist = sorted(glob(os.path.join(root_path, 'Annotation/*.txt')))
    for txtfile in txtlist:
        print(txtfile)
        # img = cv2.imread(root_path +'/PNGImages/'+ os.path.basename(txtfile[:-4]) + ".png")
        # height, width, _ = img.shape
        height, width = 0,0
        matrixs = []
        f1 = open(txtfile, 'r')
        for line in f1.readlines():
            if re.findall('Xmin', line):
                pt = [int(x) for x in re.findall(r"\d+", line)]
                matrixs.append(pt)
            elif re.findall('(X x Y x C)', line):
                height, width, _ = [int(x) for x in re.findall(r"\d+", line)]
        f1.close()
        xml = produceXMl(os.path.basename(txtfile), height, width, matrixs)

        save_xml = os.path.join('/home/lmin/data/PennFudanPed/xmls', os.path.basename(txtfile).replace('.txt', '.xml'))
        with open(save_xml, 'wb') as f:
            f.write(xml)

# txt2voc()


import xml.etree.ElementTree as ET
import os
import json

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 20180000000
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                                                                   bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)

def voc2coco():
    xml_path = '/home/lmin/data/PennFudanPed/xmls'  # 这是xml文件所在的地址
    json_file = './test.json'  # 这是你要生成的json文件
    parseXmlFiles(xml_path)  # 只需要改动这两个参数就行了
    json.dump(coco, open(json_file, 'w'))

voc2coco()
print('finished!')