# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/28 9:56
# @Author : liumin
# @File : mobilenet_v2_demo.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from compressions.quantization.custom_model import MobileNetV2
from compressions.quantizations.quant_utils import get_input_sequences, replace_quant_ops


class MobilenetV2_PTQ():
    def __init__(self):
        self.data_dir = '/home/lmin/data/hymenoptera/val'
        self.model_dir = 'ckpt/mobilenet_v2_train.pt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # weight quantization parameters
        self.w_scheme = 'mse'
        self.w_bit = 8
        self.b_bit = 8

        # activation quantization parameters
        self.a_scheme = 'mse'
        self.a_bit = 8

    def load_data(self):
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_dataset = datasets.ImageFolder(self.data_dir,
                                                  data_transform)
        dataload = torch.utils.data.DataLoader(image_dataset, batch_size=1,
                                                      shuffle=False, num_workers=4)
        return dataload


    def load_model(self):
        model = MobileNetV2('mobilenet_v2', classifier=True)
        num_ftrs = model.fc[1].in_features
        model.fc[1] = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(self.model_dir, map_location='cpu'))
        return model


    def model_accuracy(self, model, dataload):
        print('-' * 10)
        # Each epoch has a training and validation phase
        model.eval()  # Set model to evaluate mode

        running_corrects = 0
        # Iterate over data.
        for inputs, labels in dataload:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # forward
            # track history if only in train
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)

        acc = running_corrects.double() / len(dataload.dataset)
        return acc



    def fuse(self, model):
        print('step 1: model fuse and optimize.')
        layer_fuse_pairs = get_input_sequences(model)
        print(layer_fuse_pairs)
        num_bits = 8
        quant_scheme = 'mse'  # 'mse': 0.9085, 'minmax': 0.9020
        replace_quant_ops(model, num_bits, quant_scheme)



    def weight_quantize(self):
        pass


    def activation_quantize(self):
        pass


    def run(self):
        dataload = self.load_data()
        model = self.load_model()
        model.to(self.device)

        acc = self.model_accuracy(model, dataload)
        print('float model Acc: {:.4f}'.format(acc))

        model = self.fuse(model)
        acc = self.model_accuracy(model, dataload)
        print('fuse model Acc: {:.4f}'.format(acc))


if __name__ == '__main__':
    mobilenetV2_PTQ = MobilenetV2_PTQ()
    mobilenetV2_PTQ.run()
    print('done!')