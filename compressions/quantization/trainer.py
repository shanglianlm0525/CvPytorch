# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/6 18:29
# @Author : liumin
# @File : trainer.py
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import os.path as osp
from torch.optim import lr_scheduler
from torchvision.models import mobilenet_v2

from compressions.quantization.custom_model import MobileNetV2


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


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
            loss = criterion(outputs, labels)

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


def prepare_data_loaders(data_path):
    dataset = datasets.ImageFolder(os.path.join(data_path, 'train'), data_transforms['train'])
    dataset_test = datasets.ImageFolder(os.path.join(data_path, 'val'), data_transforms['val'])

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size,
        sampler=test_sampler)

    return train_loader, test_loader


if __name__ == "__main__":
    batch_size = 8
    test_batch_size = 8
    epochs = 30
    data_dir = '/home/lmin/data/hymenoptera'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


    model_ft = MobileNetV2('mobilenet_v2', classifier=True, pretrained = True)
    # model_ft = mobilenet_v2(pretrained=True)
    num_ftrs = model_ft.fc[1].in_features
    model_ft.fc[1] = nn.Linear(num_ftrs, 2)


    model_ft.to(device)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    lr_scheduler_ft = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    criterion = nn.CrossEntropyLoss()
    model_ft = train_model(model_ft, criterion, optimizer_ft, lr_scheduler_ft, num_epochs=epochs)

    val_model(model_ft, dataloaders['val'])

    torch.save(model_ft.state_dict(), 'ckpt/mobilenet_v2_train.pt')