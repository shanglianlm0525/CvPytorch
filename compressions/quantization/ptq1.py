# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/7 16:19
# @Author : liumin
# @File : ptq1.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import os.path as osp

from compressions.quantization.custom_model import MobileNetV2


def val_model(model, dataloader):
    print('-' * 10)
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Acc: {:.4f}'.format(epoch_acc))


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

if __name__ == "__main__":
    data_dir = '/home/lmin/data/hymenoptera'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model_ft = MobileNetV2('mobilenet_v2',classifier=True)
    num_ftrs = model_ft.fc[1].in_features
    model_ft.fc[1] = nn.Linear(num_ftrs, 2)
    model_ft.load_state_dict(torch.load('ckpt/mobilenet_v2_train.pt', map_location='cpu'))
    model_ft.to(device)

    model_ft.eval()
    print(list(model_ft.children()))
    val_model(model_ft, dataloaders['val'])

    model_ft.fusebn()
    val_model(model_ft, dataloaders['val'])

    num_bits = 8
    model_ft.quantize(num_bits=num_bits)
    model_ft.eval()
    print('Quantization bit: %d' % num_bits)

    direct_quantize(model_ft, dataloaders['train'])

    model_ft.freeze()

    quantize_inference(model_ft, dataloaders['val'])
