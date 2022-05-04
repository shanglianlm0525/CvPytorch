# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2022/4/30 13:46
# @Author : liumin
# @File : evison.py

from Evison import Display, show_network
from torchvision import models

# 生成我们需要可视化的网络(可以使用自己设计的网络)
network = models.efficientnet_b0(pretrained=True)

# 使用show_network这个辅助函数来看看有什么网络层(layers)
show_network(network)



# 构建visualization的对象 以及 制定可视化的网络层
visualized_layer = 'features.7.0'
display = Display(network, visualized_layer, img_size=(224, 224))  # img_size的参数指的是输入图片的大小


# 加载我们想要可视化的图片
from PIL import Image
image = Image.open('/home/lmin/data/mini-imagenet/train/n1/n0153282900000005.jpg').resize((224, 224))

# 将想要可视化的图片送入display中，然后进行保存
display.save(image)