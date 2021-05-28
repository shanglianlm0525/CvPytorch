# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/26 10:36
# @Author : liumin
# @File : qat_torch.py
import copy
import os
import random
import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse
from compressions.quantization.custom_model import MobileNetV2, fusebn
from torch.quantization import get_default_qconfig, quantize_jit
import sys

# # Setup warnings
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

def qat_model(model, dataloaders,dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, use_qat=False):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            scheduler.step()

        epoch_loss = running_loss / dataset_sizes
        epoch_acc = running_corrects.double() / dataset_sizes

        print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if use_qat:
            if epoch > 3:
                # Freeze quantizer parameters
                model.apply(torch.quantization.disable_observer)
            if epoch > 2:
                # Freeze batch norm mean and variance estimates
                model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    return model


def val_model(model, dataloader):
    print('-' * 10)
    # Each epoch has a training and validation phase
    model.eval()   # Set model to evaluate mode

    running_corrects = 0
    # Iterate over data.
    for inputs, labels in dataloader:
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

        running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    print('Acc: {:.4f}'.format(epoch_acc))


def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def arguments():
    parser = argparse.ArgumentParser(description='Cross Layer Equalization in MV2')

    parser.add_argument('--images-dir', help='Imagenet eval image', default='/home/lmin/data/imagenet', type=str)
    parser.add_argument('--seed', help='Seed number for reproducibility', type=int, default=0)
    parser.add_argument('--ptq',
                        help='Post Training Quantization techniques to run - Select from CLS / HBA / Bias correction',
                        nargs='+', default=[None])
    parser.add_argument('--quant-scheme', help='Quantization scheme', default='mse', type=str,
                        choices=['mse', 'minmax'])

    parser.add_argument('--batch-size', help='Data batch size for a model', type=int, default=64)
    parser.add_argument('--num-workers', help='Number of workers to run data loader in parallel', type=int, default=16)

    args = parser.parse_args()
    return args


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

if __name__ == '__main__':
    args = arguments()
    seed(args)

    data_dir = '/home/lmin/data/hymenoptera'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    model = MobileNetV2('mobilenet_v2', classifier=True)
    num_ftrs = model.fc[1].in_features
    model.fc[1] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('ckpt/mobilenet_v2_train.pt', map_location='cpu'))
    # model.to(device)
    model.to('cpu')
    model.eval()

    val_model(model, dataloaders['val'])

    model = fusebn(model)
    val_model(model, dataloaders['val'])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    lr_scheduler1 = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

    torch.quantization.prepare_qat(model, inplace=True)
    print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n')

    model.to(device)
    model = qat_model(model, dataloaders['val'], dataset_sizes['val'], criterion, optimizer, lr_scheduler1, 25, use_qat=True)

    # Check the accuracy after each epoch
    model.to('cpu')
    quantized_model = torch.quantization.convert(model.eval(), inplace=False)
    quantized_model.eval()
    val_model(quantized_model, dataloaders['val'])

