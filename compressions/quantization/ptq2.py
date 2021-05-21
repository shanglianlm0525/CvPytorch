# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/5/20 16:16
# @Author : liumin
# @File : ptq2.py
import os
import random
import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import argparse

from compressions.quantization.custom_model import MobileNetV2, replace_quant_ops, fusebn, run_calibration, \
    set_quant_mode
from compressions.quantization.quant_ops import QuantConv


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


def seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def get_input_sequences(model):
    layer_bn_pairs = []

    def hook(name):
        def func(m, i, o):
            if m in (torch.nn.Conv2d, torch.nn.Linear):
                if not layer_bn_pairs:
                    layer_bn_pairs.append((m, name))
                else:
                    if layer_bn_pairs[-1][0] in (torch.nn.Conv2d, torch.nn.Linear):
                        layer_bn_pairs.pop()
            else:
                layer_bn_pairs.append((m, name))

        return func

    handlers = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            handlers.append(module.register_forward_hook(hook(name)))
    dummy = torch.randn([1, 3, 224, 224]).cuda()
    model(dummy)
    for handle in handlers:
        handle.remove()
    return layer_bn_pairs


def register_bn_params_to_prev_layers(model, layer_bn_pairs):
    idx = 0
    while idx + 1 < len(layer_bn_pairs):
        conv, bn = layer_bn_pairs[idx], layer_bn_pairs[idx + 1]
        conv, conv_name = conv
        bn, bn_name = bn
        bn_state_dict = bn.state_dict()
        conv.register_buffer('eps', torch.tensor(bn.eps))
        conv.register_buffer('gamma', bn_state_dict['weight'].detach())
        conv.register_buffer('beta', bn_state_dict['bias'].detach())
        conv.register_buffer('mu', bn_state_dict['running_mean'].detach())
        conv.register_buffer('var', bn_state_dict['running_var'].detach())
        idx += 2


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
    model.to(device)
    val_model(model, dataloaders['val'])


    layer_bn_pairs = get_input_sequences(model)
    register_bn_params_to_prev_layers(model, layer_bn_pairs)

    def bn_fold(module):
        if isinstance(module, (QuantConv)):
            module.batchnorm_folding()

    ## 2 replace_quant_ops
    num_bits = 8
    quant_scheme = 'mse'  # 'minmax'
    replace_quant_ops(model, num_bits, quant_scheme)
    model.apply(bn_fold)
    val_model(model, dataloaders['val'])

    model.apply(run_calibration(calibration = True))
    val_model(model, dataloaders['val'])

    # replace_quant_to_brecq_quant(model)
    model.apply(set_quant_mode(quantized = True))
    val_model(model, dataloaders['val'])

