# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/5/5 15:52
# @Author : liumin
# @File : WiderFaceProc.py

import os, cv2, sys, shutil
from xml.dom.minidom import Document


def writexml(filename, saveimg, bboxes, xmlpath):  # 定义写xml文件的函数
    doc = Document()  # 定义一个文件
    annotation = doc.createElement('annotation')  # 定义一个结点annotation
    doc.appendChild(annotation)  # 将根结点annotation作为doc的一个子节点
    folder = doc.createElement('folder')  # 定义一个节点folder
    folder_name = doc.createTextNode('widerface')
    folder.appendChild(folder_name)
    annotation.appendChild(folder)  # 将folder作为annotation的子节点
    filenamenode = doc.createElement('filename')  # 定义一个节点filenamenode
    filename_name = doc.createTextNode(filename)
    filenamenode.appendChild(filename_name)
    annotation.appendChild(filenamenode)  # 将filenamenode作为annotation的子节点
    source = doc.createElement('source')  # 定义一个节点source
    annotation.appendChild(source)  # 将source作为annotation的子节点
    database = doc.createElement('database')  # 定义一个节点database
    database.appendChild(doc.createTextNode('WiderFace'))
    source.appendChild(database)
    annotation_s = doc.createElement('annotation')
    annotation_s.appendChild(doc.createTextNode('PASCAL'))
    source.appendChild(annotation_s)
    image = doc.createElement('image')
    image.appendChild(doc.createTextNode('flickr'))
    source.appendChild(image)
    flickrid = doc.createElement('flickrid')
    flickrid.appendChild(doc.createTextNode('-1'))
    source.appendChild(flickrid)
    size = doc.createElement('size')  # 定义一个子节点size
    annotation.appendChild(size)
    width = doc.createElement('width')
    width.appendChild(doc.createTextNode(str(saveimg.shape[1])))
    height = doc.createElement('height')
    height.appendChild(doc.createTextNode(str(saveimg.shape[0])))
    depth = doc.createElement('depth')
    depth.appendChild(doc.createTextNode(str(saveimg.shape[2])))
    size.appendChild(width)
    size.appendChild(height)
    size.appendChild(depth)
    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        objects = doc.createElement('object')
        annotation.appendChild(objects)
        object_name = doc.createElement('name')
        object_name.appendChild(doc.createTextNode('face'))
        objects.appendChild(object_name)
        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        objects.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('0'))
        objects.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        objects.appendChild(difficult)
        bndbox = doc.createElement('bndbox')
        objects.appendChild(bndbox)
        xmin = doc.createElement('xmin')
        xmin.appendChild(doc.createTextNode(str(bbox[0])))
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin.appendChild(doc.createTextNode(str(bbox[1])))
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax.appendChild(doc.createTextNode(str(bbox[0] + bbox[2])))
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax.appendChild(doc.createTextNode(str(bbox[1] + bbox[3])))
        bndbox.appendChild(ymax)
    f = open(xmlpath, "w")
    f.write(doc.toprettyxml(indent=''))
    f.close()


rootdir = "/home/lmin/data/WiderFace"  # 定义根目录


# 解析真值文件的函数
def convertimgset(img_set):
    # 指向图像，有两个路径，可能是train也可能是val
    imgdir = rootdir + "/WIDER_" + img_set + "/images"
    # 指向真值信息
    gtfilepath = rootdir + "/wider_face_split/wider_face_" + img_set + "_bbx_gt.txt"
    fwrite = open(rootdir + "/" + img_set + ".txt", 'w')  # 定义一个文件对象，用来存放图片目录
    # 定义一个索引
    if img_set == 'train':
        index = 12880
    elif img_set == 'val':
        index = 3226
    else:
        index = 0
    # 打开真值文件
    with open(gtfilepath, 'r') as gtfiles:
        while index:
            filename = gtfiles.readline()[:-1]  # 读取图像路径，第一行是图片的路径
            if filename == None or filename == "":  # 判断一下文件的路径是否存在
                #############
                continue  # 如果不存在就继续读取
            imgpath = imgdir + "/" + filename  #
            img = cv2.imread(imgpath)  # 用opencv读取图片，用到后面的size里
            if not img.data:  # 判断图片是否存在
                break  # 如果不存在就报错
            # numbbox = int(gtfiles.readline())
            numbbox = max(1, int(gtfiles.readline()))  # 读入图片中的人脸数（第二行）,这里如果使用上面那行代码就会报错
            # 因为会有人脸框为0的图片，会导致下面的循环出错
            # 改成1可以才能把人脸框的数据读进来，尽管他是0 0 0 0
            bboxes = []
            # print(numbbox)
            for i in range(numbbox):
                line = gtfiles.readline()  # 继续读下一行，每一行就是一个人脸框
                lines = line.split(" ")  # 按空格对信息进行分割
                lines = lines[0:4]  # 前四个就是标注信息，取0123，分别是人脸框左上角的坐标值，和框的高宽
                # 将标注信息存放到bounding box中，左上角的坐标值，和宽度，高度
                bbox = (int(lines[0]), int(lines[1]), int(lines[2]), int(lines[3]))
                # if int(lines[2]) < 40 or int(lines[3]) < 40:
                #    continue
                # 将bounding box添加在一个集合中
                bboxes.append(bbox)
                # cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),color=(255,255,0),thickness=1)
            # 定义图片的名字，把/转化为_，将文件夹和文件名合并
            # filename = filename.replace("/", "_")
            # 判断是否存在标注信息
            if len(bboxes) == 0:  # bboxs的长度为0的话就代表没有收集到人脸
                print("no face")  # 这时候输出no face
                continue  # 没有人脸就不执行下面的操作了，没有人脸就不把这张图的名字写入txt，也不打印成功信息

            fwrite.write(
                'WIDER_' + img_set + '/images/' + filename + ' ' + 'WIDER_' + img_set + '/annotations/' + filename.replace(
                    ".jpg", ".xml") + '\n')  # 将当前图片的名字写入到txt文件中
            xmlpath = "{}/WIDER_{}/annotations/{}.xml".format(rootdir, img_set, filename.split(".")[0])  # 定义xml的存储路径
            if not os.path.isdir(os.path.dirname(xmlpath)):
                os.makedirs(os.path.dirname(xmlpath))
            writexml(filename, img, bboxes, xmlpath)  # 调用上面的函数将相应的参数传入到指定的位置
            index = index - 1
            print(index)
    fwrite.close()  # 关闭文件对象


if __name__ == "__main__":  # 定义一个main函数
    img_sets = ["train", "val"]  # 定义需要解析的文件的集合
    for img_set in img_sets:  # 循环遍历这两个文件
        convertimgset(img_set)  # 调用函数
