# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/19 14:19
# @Author : liumin
# @File : dali_transforms.py

from nvidia.dali.pipeline import Pipeline
# help(Pipeline)
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os.path
import fnmatch


class SimplePipeline(Pipeline):
    def __init__(self, image_dir='/home/lmin/data/hymenoptera/train/ants', batch_size=8, num_threads=1, device_id=0):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)


pipe = SimplePipeline()
pipe.build()
pipe_out = pipe.run()
print(pipe_out)