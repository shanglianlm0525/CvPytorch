# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/10 16:24
# @Author : liumin
# @File : trainer.py

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import lr_scheduler
import torchvision
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms as transformsT
from torchvision import datasets as datasetsT
from torchvision import models as modelsT

import attr
from tqdm import tqdm
import logging
from copy import deepcopy
import importlib
from time import time
from datetime import datetime
from pathlib import Path as P
from math import ceil
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path as P

from src.lr_schedulers.lr_scheduler import parser_lr_scheduler
from src.optimizers.optimizer import parser_optimizer
from src.utils.config import CommonConfiguration
from src.utils.logger import logger

from src.utils.timer import Timer
from src.utils.metrics import LossMeter,PerfMeter
from src.utils.tensorboard import DummyWriter
from src.utils.checkpoints import Checkpoints


cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.enabled = True

def get_current_lr(optimizer):
    return min(g["lr"] for g in optimizer.param_groups)

def get_class_name(full_class_name):
    return full_class_name.split(".")[-1]


def clip_grad(cfg, model):
    if cfg.GRAD_CLIP.TYPE == "norm":
        clip_method = clip_grad_norm_
    elif cfg.GRAD_CLIP.TYPE == "value":
        clip_method = clip_grad_value_
    else:
        raise NotImplementedError

    clip_method(model.parameters(), cfg.GRAD_CLIP.VALUE)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def prepare_transforms():
    data_transforms = {
        'train': transformsT.Compose([
            transformsT.RandomResizedCrop(224),
            transformsT.RandomHorizontalFlip(),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'val': transformsT.Compose([
            transformsT.Resize(256),
            transformsT.CenterCrop(224),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'test': transformsT.Compose([
            transformsT.Resize(256),
            transformsT.CenterCrop(224),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

class Relabel:
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.FloatTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor

class ToLabel255:
    def __call__(self, image):
        return torch.from_numpy(np.array(image)/255).float().unsqueeze(0)

class ToLabel:
    def __call__(self, image):
        return torch.from_numpy(image).float().unsqueeze(0)

def prepare_transforms_seg():
    data_transforms = {
        'train': transformsT.Compose([
            transformsT.Resize((800,600)),
            # transformsT.RandomHorizontalFlip(),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'val': transformsT.Compose([
            transformsT.Resize((800,600)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'test': transformsT.Compose([
            transformsT.Resize((800,600)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def prepare_transforms_mask():
    data_transforms = {
        'train': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ]),

        'val': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ]),

        'test': transformsT.Compose([
            # transformsT.Resize((800,600)),
            ToLabel(),
        ])
    }
    return data_transforms

def prepare_transforms_seg_cityscapes():
    data_transforms = {
        'train': transformsT.Compose([
            transformsT.Resize((640,320)),
            # transformsT.RandomHorizontalFlip(),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'val': transformsT.Compose([
            transformsT.Resize((640,320)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),

        'test': transformsT.Compose([
            transformsT.Resize((640,320)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def prepare_transforms_mask_cityscapes():
    data_transforms = {
        'train': transformsT.Compose([
            # transformsT.Resize((640,320)),
            ToLabel(),
        ]),

        'val': transformsT.Compose([
            # transformsT.Resize((640,320)),
            ToLabel(),
        ]),

        'test': transformsT.Compose([
            # transformsT.Resize((640,320)),
            ToLabel(),
        ])
    }
    return data_transforms

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_epoch = -1
        self.n_iters_elapsed = 0
        self.device = self.cfg.GPU_IDS
        self.batch_size = self.cfg.BATCH_SIZE
        self.batch_size_all = self.cfg.BATCH_SIZE * len(self.cfg.GPU_IDS)

        self.n_steps_per_epoch = None
        self.logger = logging.getLogger("pytorch")
        self.experiment_id = self.experiment_id(self.cfg)
        self.checkpoints = Checkpoints(self.logger,self.cfg.CHECKPOINT_DIR,self.experiment_id)
        self.tb_writer = DummyWriter(log_dir="%s/%s" % (self.cfg.TENSORBOARD_LOG_DIR, self.experiment_id))

    def experiment_id(self, cfg):
        return f"{cfg.EXPERIMENT_NAME}#{cfg.USE_MODEL.split('.')[-1]}#{cfg.OPTIMIZER.TYPE}#{cfg.LR_SCHEDULER.TYPE}" \
               f"#{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    def _parser_dict(self):
        dictionary = CommonConfiguration.from_yaml(cfg.DATASET.DICTIONARY)
        return next(dictionary.items())[1] ## return first

    def _parser_datasets(self,dictionary):
        transforms = prepare_transforms_seg()
        target_transforms = prepare_transforms_mask()
        *dataset_str_parts, dataset_class_str = cfg.DATASET.CLASS.split(".")
        dataset_class = getattr(importlib.import_module(".".join(dataset_str_parts)), dataset_class_str)
        datasets = {x: dataset_class(data_cfg=cfg.DATASET[x.upper()], dictionary=dictionary,transform=transforms[x],target_transform=target_transforms[x], stage=x) for x in ['train', 'val']}
        dataloaders = {x: DataLoader(datasets[x], batch_size=self.batch_size_all, num_workers=cfg.NUM_WORKERS,
                                     shuffle=cfg.DATASET[x.upper()].SHUFFLE,   pin_memory=True) for x in ['train', 'val']} # collate_fn=detection_collate,
        return datasets, dataloaders


    def _parser_model(self,dictionary):
        *model_mod_str_parts, model_class_str = self.cfg.USE_MODEL.split(".")
        model_class = getattr(importlib.import_module(".".join(model_mod_str_parts)), model_class_str)
        model = model_class(dictionary=dictionary)
        return model

    def run(self):
        cfg = self.cfg
        # cfg.print()

        ## parser_dict
        dictionary = self._parser_dict()

        ## parser_datasets
        datasets, dataloaders = self._parser_datasets(dictionary)
        # dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
        # class_names = datasets['train'].classes

        ## parser_model
        model_ft = self._parser_model(dictionary)

        ## parser_optimizer
        optimizer_ft = parser_optimizer(cfg,model_ft)


        ## parser_lr_scheduler
        lr_scheduler_ft = parser_lr_scheduler(cfg, optimizer_ft)


        if self.cfg.PRETRAIN_MODEL is not None:
            if self.cfg.RESUME:
                self.start_epoch = self.checkpoints.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft, optimizer_ft, lr_scheduler_ft)
            else:
                self.checkpoints.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft)

        if torch.cuda.is_available():
            model_ft = model_ft.cuda()
            cudnn.benchmark = True
            for state in optimizer_ft.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        ## vis net graph
        if self.cfg.TENSORBOARD_MODEL and False:
            self.tb_writer.add_graph(model_ft, (model_ft.dummy_input.cuda(),))

        if self.cfg.HALF:
            model_ft.half()

        self.n_steps_per_epoch = int(ceil(sum(len(t) for t in datasets['train'])))

        best_acc = 0.0
        for epoch in range(self.start_epoch + 1, self.cfg.N_MAX_EPOCHS):
            self.train_epoch(epoch, model_ft, dataloaders['train'], optimizer_ft, lr_scheduler_ft,None)
            if self.cfg.DATASET.VAL:
                acc = self.val_epoch(epoch, model_ft, dataloaders['val'] , optimizer=optimizer_ft, lr_scheduler=lr_scheduler_ft)
                # start to save best performance model after learning rate decay to 1e-6
                if best_acc < acc:
                    self.checkpoints.autosave_checkpoint(model_ft, epoch, 'best', optimizer_ft, lr_scheduler_ft)
                    best_acc = acc
                    continue

            if not epoch % cfg.N_EPOCHS_TO_SAVE_MODEL:
                self.checkpoints.autosave_checkpoint(model_ft, epoch,'autosave', optimizer_ft, lr_scheduler_ft)

        self.tb_writer.close()


    def train_epoch(self, epoch, model, dataloader, optimizer, lr_scheduler, grad_normalizer=None, prefix="train"):
        model.train()

        _timer = Timer()
        lossMeter = LossMeter()
        perfMeter = PerfMeter()

        for i, (imgs, labels) in enumerate(dataloader):
            _timer.tic()
            # zero the parameter gradients
            optimizer.zero_grad()

            if self.cfg.HALF:
                imgs = imgs.half()

            if len(self.device)>1:
                out = data_parallel(model, (imgs, labels, prefix), device_ids=self.device, output_device=self.device[0])
            else:
                imgs = imgs.cuda()
                labels = [label.cuda() for label in labels] if isinstance(labels,list) else labels.cuda()
                out = model(imgs, labels, prefix)

            if not isinstance(out, tuple):
                losses, performances = out, None
            else:
                losses, performances = out

            if losses["all_loss"].sum().requires_grad:
                if self.cfg.GRADNORM is not None:
                    grad_normalizer.adjust_losses(losses)
                    grad_normalizer.adjust_grad(model, losses)
                else:
                    losses["all_loss"].sum().backward()

            optimizer.step()

            self.n_iters_elapsed += 1

            _timer.toc()

            lossMeter.__add__(losses)

            if performances is not None and all(performances):
                perfMeter.put(performances)

            if (i + 1) % self.cfg.N_ITERS_TO_DISPLAY_STATUS == 0:
                avg_losses = lossMeter.average()
                template = "[epoch {}/{}, iter {}, lr {}] Total train loss: {:.4f} " "(ips = {:.2f} )\n" "{}"
                self.logger.info(
                    template.format(
                        epoch, self.cfg.N_MAX_EPOCHS, i,
                        round(get_current_lr(optimizer), 6),
                        avg_losses["all_loss"],
                        self.batch_size * self.cfg.N_ITERS_TO_DISPLAY_STATUS /  _timer.total_time,
                        "\n".join(["{}: {:.4f}".format(n, l) for n, l in avg_losses.items() if n != "all_loss"]),
                    )
                )

                if self.cfg.TENSORBOARD:
                    tb_step = int((epoch * self.n_steps_per_epoch + i) / self.cfg.N_ITERS_TO_DISPLAY_STATUS)
                    # Logging train losses
                    [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l, tb_step) for n, l in avg_losses.items()]

                lossMeter.clear()

            del imgs, labels, losses, performances

        lr_scheduler.step()

        if self.cfg.TENSORBOARD and len(perfMeter):
            avg_perf = perfMeter.average()
            [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v, epoch) for k, v in avg_perf.items()]

        if self.cfg.TENSORBOARD_WEIGHT and False:
            for name, param in model.named_parameters():
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                self.tb_writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    @torch.no_grad()
    def val_epoch(self, epoch, model, dataloader, optimizer=None, lr_scheduler=None, prefix="val"):
        model.eval()

        lossMeter = LossMeter()
        perfMeter = PerfMeter()

        with torch.no_grad():
            for (imgs, labels) in dataloader:

                if self.cfg.HALF:
                    imgs = imgs.half()

                if len(self.device)>1:
                    losses, performances = data_parallel(model, (imgs, labels, prefix), device_ids=self.device,
                                                         output_device=self.device[-1])
                else:
                    imgs = imgs.cuda()
                    labels = [label.cuda() for label in labels] if isinstance(labels,list) else labels.cuda()
                    losses, performances = model(imgs, labels, prefix)

                lossMeter.__add__(losses)
                perfMeter.__add__(performances)

                del imgs, labels, losses, performances

        avg_losses = lossMeter.average()
        avg_perf = perfMeter.average()

        template = "[epoch {}] Total {} loss : {:.4f} " "\n" "{}"
        self.logger.info(
            template.format(
                epoch,prefix,avg_losses["all_loss"],
                "\n".join(["{}: {:.4f}".format(n, l) for n, l in avg_losses.items() if n != "all_loss"]),
            )
        )

        if self.cfg.TENSORBOARD:
            # Logging val Loss
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l, epoch) for n, l in avg_losses.items()]
            # Logging val performances
            [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v, epoch) for k, v in avg_perf.items()]

        perf_log_str = f"\n------------ Performances ({prefix}) ----------\n"
        for k,v in avg_perf.items():
            perf_log_str += "{:}: {:.4f}\n".format(k, v)
        perf_log_str += "------------------------------------\n"
        self.logger.info(perf_log_str)

        acc = avg_perf['all_perf']

        del avg_losses, avg_perf
        return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
    parser.add_argument('--setting', default='conf/cityscapes.yml', help='The path to the configuration file.')

    args = parser.parse_args()
    cfg = CommonConfiguration.from_yaml(args.setting)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, cfg.GPU_IDS)))

    logger.info('Loaded configuration file: {}'.format(args.setting))
    logger.info('Use gpu ids: {}'.format(cfg.GPU_IDS))

    trainer = Trainer(cfg)
    logger.info('Begin to training ...')
    trainer.run()
    logger.info('finish!')
    torch.cuda.empty_cache()