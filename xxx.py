# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/9/15 16:32
# @Author : liumin
# @File : xxx.py
import bisect
import copy
import pickle
from collections import OrderedDict

import cv2
import math
import torch
import torch.nn as nn
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

from src.evaluator import SegmentationEvaluator


def make_onehot_test():
    x = (torch.rand(2,1,3,4)*10).int()
    print(x)
    
    masks_onehot = torch.zeros_like(x)
    for i in range(1,10):
        masks_onehot[:,x == i,:,:] = 1
    
    print(masks_onehot)
    masks = torch.zeros(3, 4)
    for ipt in x:
        for it in ipt:
            for i in range(1,10):
                masks.fill_(0)
                print(i)
                masks[it==i] = 1
                print(masks)


def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0 , 0, im.shape[1] - 1)
    x1 = np.clip(x1, 0, im.shape[1] - 1)
    y0 = np.clip(y0, 0, im.shape[0] - 1)
    y1 = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa =(x1 - x) * (y1 - y)
    wb =(x1 - x) * (y - y0)
    wc =(x - x0 ) * (y1 - y)
    wd =(x - x0) * (y - y0)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def testCE():
    x_input = torch.randn(3, 3)  # 随机生成输入
    y_target = torch.tensor([1, 2, 0])  # 设置输出具体值 print('y_target\n',y_target)
    crossentropyloss = nn.CrossEntropyLoss()
    crossentropyloss_output = crossentropyloss(x_input, y_target)
    print('x_input',x_input)
    print('y_target',y_target)
    print('crossentropyloss_output',crossentropyloss_output)

    softmax_func = nn.Softmax(dim=1)
    soft_output = softmax_func(x_input)
    print('soft_output:\n', soft_output)

    # 在softmax的基础上取log
    log_output = torch.log(soft_output)
    print('log_output:\n', log_output)

    nllloss_func = nn.NLLLoss()
    nlloss_output = nllloss_func(log_output, y_target)
    print('nlloss_output:\n', nlloss_output)


def testCeAndBce():
    x_input = torch.randn(4, 3, 3, 3)  # 随机生成输入
    y_target = torch.randint(0,3,(4,1,3,3))
    celoss = nn.CrossEntropyLoss()
    ce_output = celoss(x_input, y_target.squeeze(dim=1).long())
    print('x_input', x_input)
    print('y_target', y_target)
    print('ce_output', ce_output)

    bceloss = nn.BCEWithLogitsLoss()
    loss_list = []
    for idx in range(3):
        y_labels_onehot = torch.zeros_like(y_target)
        y_labels_onehot[y_target == idx] = 1
        loss_list.append(bceloss(x_input[:,idx,:,:].unsqueeze(dim=1), y_labels_onehot.float()))
    print('loss_list',loss_list,np.mean(loss_list),np.sum(loss_list))



# testCeAndBce()

def testTensor():
    toTensor = ToTensor()

    mask = Image.open('/home/lmin/data/portrait/train/masks/00001_matte.png')
    mask = np.array(mask, dtype=np.uint8)
    # Put all non-void classes to zero
    print(mask)
    mask[mask ==255] = 1
    print(mask)
    mask_np = torch.from_numpy(mask)
    print(mask_np)
    mask = Image.fromarray(mask)
    mask_image = toTensor(mask)
    print(mask_image)

## testTensor()

color_encoding = OrderedDict([
    ('sky', (128, 128, 128)),
    ('building', (128, 0, 0)),
    ('pole', (192, 192, 128)),
    ('road_marking', (255, 69, 0)),
    ('road', (128, 64, 128)),
    ('pavement', (60, 40, 222)),
    ('tree', (128, 128, 0)),
    ('sign_symbol', (192, 128, 128)),
    ('fence', (64, 64, 128)),
    ('car', (64, 0, 128)),
    ('pedestrian', (64, 64, 0)),
    ('bicyclist', (0, 128, 192)),
    ('unlabeled', (0, 0, 0))
])

color_encoding_id = OrderedDict([
    (0, (128, 128, 128)),
    (1, (128, 0, 0)),
    (2, (192, 192, 128)),
    (3, (255, 69, 0)),
    (4, (128, 64, 128)),
    (5, (60, 40, 222)),
    (6, (128, 128, 0)),
    (7, (192, 128, 128)),
    (8, (64, 64, 128)),
    (9, (64, 0, 128)),
    (10, (64, 64, 0)),
    (11, (0, 128, 192)),
    (12, (0, 0, 0))
])

def checkCamvid():
    img = cv2.imread('wiki/imgs/Camvid_Mask.png')
    mask = cv2.imread('wiki/imgs/Camvid_Img.png', 0)

    show_mask = np.zeros_like(img)
    for idx in range(12):
        show_mask_tmp = np.zeros_like(img)
        show_mask[mask==idx] = color_encoding_id[idx]
        show_mask_tmp[mask==idx] = color_encoding_id[idx]
        cv2.imshow('show_mask_tmp_'+str(idx), show_mask_tmp)

    cv2.imshow('img', img)
    cv2.imshow('mask', mask)
    cv2.imshow('show_mask', show_mask)
    cv2.waitKey(0)


# checkCamvid()


import time
import asyncio

async def hello(i):
    print('Hello World:%s' % time.time())
    return i

def run():
    loop = asyncio.get_event_loop()
    tasks = []
    for i in range(5):
        task = loop.create_task(hello(i))
        loop.run_until_complete(task)
        tasks.append(task.result())
    print([task for task in tasks])

def run_gather():
    loop = asyncio.get_event_loop()
    coros = []
    for i in range(5):
        task = loop.create_task(hello(i))
        coros.append(task)
    loop.run_until_complete(asyncio.gather(*coros))
    print([coro.result() for coro in coros])

# run()
# run_gather()


import matplotlib.pyplot as plt


def testLR(epochs=300):
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
    print(lf)
    y = []
    for x in range(epochs):
        y.append(lf(x))

    x = np.linspace(0, epochs-1, epochs)
    plt.plot(x, y)
    plt.show()

# testLR()

def luoXuan(n=3, m=4):
    arr = np.zeros((n,m), dtype=np.int)

    for i in range(n):
        for j in range(m):
            pass



# luoXuan()

def testBCE():
    target = torch.ones([8,4, 5, 7], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.ones([8,4, 5, 7])  # A prediction (logit)
    # pos_weight = torch.ones([6])  # All weights are equal to 1
    criterion = nn.BCEWithLogitsLoss()
    criterion(output, target)  # -log(sigmoid(1.5))
    print('bce_loss', output.shape)
    print('bce_loss', output)


# testBCE()


def testBCEReduce():
    target = torch.ones([8, 4, 5, 7], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.ones([8, 4, 5, 7])  # A prediction (logit)

    criterion1 = nn.BCEWithLogitsLoss(reduction='sum')
    loss1 = criterion1(output, target)  # -log(sigmoid(1.5))
    print('loss1', loss1)

    criterion2 = nn.BCEWithLogitsLoss(reduction='mean')
    loss2 = criterion2(output, target)  # -log(sigmoid(1.5))
    print('loss2', loss2)

    criterion3 = nn.BCEWithLogitsLoss(reduction='none')
    loss3 = criterion3(output, target)  # -log(sigmoid(1.5))
    print('loss3', loss3)


# testBCEReduce()


def test_flashtorch():
    from torchvision import models
    from flashtorch.activmax  import gradient_ascent
    from flashtorch.utils import imagenet, apply_transforms, load_image
    from flashtorch.saliency import Backprop
    from flashtorch import utils


    model = models.alexnet(pretrained=True)

    backprop = Backprop(model)

    img = load_image('/home/lmin/data/imagenet/great_grey_owl.jpg')
    img = apply_transforms(img)

    target_class = 24

    backprop.visualize(img, target_class, guided=True)


def test_aspect_ratios():
    def _quantize(x, bins):
        bins = copy.deepcopy(bins)
        bins = sorted(bins)
        quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
        return quantized

    k = 3
    aspect_ratios = [1.2,  2, 3 , 0.9, 0.2, 1.8,  2.3, 3.9, 1.1, 1.2, 0.1,  2.3 , 0.75 , 1.1, 1]
    bins = (2 ** np.linspace(-1, 1, 2 * k + 1)).tolist()
    groups = _quantize(aspect_ratios, bins)
    print('groups', groups)
    # count number of elements per group
    counts = np.unique(groups, return_counts=True)[1]
    print('counts', counts)
    fbins = [0] + bins + [np.inf]
    print("Using {} as bins for aspect ratio quantization".format(fbins))
    print("Count of instances per bin: {}".format(counts))


def test_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)
    print(predictions)


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        acc = torch.diag(h) / h.sum(1)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

def test_seg_eval():
    dataset = torch.load("dataset.pt")
    gt_list = torch.load("gt_list.pt")
    predict_list = torch.load("predict_list.pt")
    print()

    eval1 = SegmentationEvaluator(dataset)
    eval2 = ConfusionMatrix(dataset.num_classes)

    for gt,predict in zip(gt_list, predict_list):
        eval1.update(gt,predict)
        eval2.update(gt.flatten(), predict.flatten())

    performances = eval1.evaluate()
    print('performances', performances)

    print('confmat', eval2)




# test_seg_eval()

# n_threads = torch.get_num_threads()
# print(n_threads)

from torchcontrib.optim.swa import SWA

model = torchvision.models.mobilenet_v2()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001
)

optimizer = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)


def get_current_lr(optimizer):
    return min(g["lr"] for g in optimizer.param_groups)

for i in range(100):
    optimizer.step()
    print(get_current_lr(optimizer))



