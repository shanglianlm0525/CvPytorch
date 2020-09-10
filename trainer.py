# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/10 16:24
# @Author : liumin
# @File : trainer.py

import argparse
# import logging
import os
from collections import defaultdict

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import lr_scheduler
import torchvision
from torch.nn.parallel import data_parallel
# from torch.nn import SyncBatchNorm
# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
from torch.utils.data.dataloader import default_collate
from torchvision import transforms as transformsT
from torchvision import datasets as datasetsT
from torchvision import models as modelsT

import attr
from tqdm import tqdm
from copy import deepcopy
import importlib
from time import time
from datetime import datetime
from pathlib import Path as P
from math import ceil
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path as P
import torch.distributed as dist

try:
    import apex
    from apex import amp
    from apex.parallel import DistributedDataParallel, SyncBatchNorm, convert_syncbn_model
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex.")

from src.utils.distributed import torch_distributed_zero_first
from src.lr_schedulers.lr_scheduler import parser_lr_scheduler
from src.optimizers.optimizer import parser_optimizer
from src.utils.config import CommonConfiguration
from src.utils.logger import logger
from src.utils.timer import Timer
from src.utils.metrics import LossMeter,PerfMeter
from src.utils.tensorboard import DummyWriter
from src.utils.checkpoints import Checkpoints
from src.utils.distributed import init_distributed,is_main_process, reduce_dict, MetricLogger

cudnn.benchmark = False
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

from torch._six import container_abcs, string_classes, int_classes

def default_collate1(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        if elem_type.__name__ == 'tuple':
            if isinstance(elem[1],torch.Tensor):
                pass
            elif type(batch[1]).__name__ == 'tuple': # if type(batch[1]).__name__ == 'tuple':
                return tuple(zip(*batch))
            else:
                targets = []
                imgs = []
                for sample in batch:
                    imgs.append(sample[0])
                    targets.append(torch.FloatTensor(sample[1])) # torch.FloatTensor(sample[1])
                return torch.stack(imgs, 0), targets
        else:
            transposed = zip(*batch)
            return [default_collate(samples) for samples in transposed]
    raise TypeError(default_collate_err_msg_format.format(elem_type))

def default_collate2(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

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

# logger = logging.getLogger("pytorch")

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.start_epoch = -1
        self.n_iters_elapsed = 0
        self.device = self.cfg.GPU_IDS
        self.batch_size = self.cfg.BATCH_SIZE
        self.batch_size_all = self.cfg.BATCH_SIZE * len(self.cfg.GPU_IDS)

        self.n_steps_per_epoch = None
        if cfg.local_rank == 0:
            self.experiment_id = self.experiment_id(self.cfg)
            self.ckpts = Checkpoints(logger,self.cfg.CHECKPOINT_DIR,self.experiment_id)
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

        datasets = {x: dataset_class(data_cfg=cfg.DATASET[x.upper()], dictionary=dictionary, transform=transforms[x],
                                     target_transform=target_transforms[x], stage=x) for x in ['train', 'val']}

        data_samplers = defaultdict()
        if self.cfg.distributed:
            data_samplers = {x: torch.utils.data.distributed.DistributedSampler(datasets[x],shuffle=cfg.DATASET[x.upper()].SHUFFLE) for x in ['train', 'val']}
        else:
            data_samplers['train'] = torch.utils.data.RandomSampler(datasets['train'])
            data_samplers['val'] = torch.utils.data.SequentialSampler(datasets['val'])

        dataloaders = {x: DataLoader(datasets[x], batch_size=self.batch_size,sampler=data_samplers[x], num_workers=cfg.NUM_WORKERS,
                                      collate_fn=default_collate2, pin_memory=True) for x in ['train', 'val']} # collate_fn=detection_collate,

        return datasets, dataloaders, data_samplers


    def _parser_model(self,dictionary):
        *model_mod_str_parts, model_class_str = self.cfg.USE_MODEL.split(".")
        model_class = getattr(importlib.import_module(".".join(model_mod_str_parts)), model_class_str)
        model = model_class(dictionary=dictionary,device=cfg.device)
        return model

    def run(self):
        cfg = self.cfg
        # cfg.print()

        ## init distributed
        # self.distributed = init_distributed(dist_backend = 'nccl',dist_url='env://',world_size=1)
        # self.distributed = True if cfg.rank != -1 else False

        ## parser_dict
        dictionary = self._parser_dict()

        ## parser_datasets
        datasets, dataloaders,data_samplers = self._parser_datasets(dictionary)
        # dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
        # class_names = datasets['train'].classes

        ## parser_model
        model_ft = self._parser_model(dictionary)

        if cfg.distributed and False:
            model_ft = apex.parallel.convert_syncbn_model(model_ft).cuda()
        else:
            model_ft = model_ft.cuda()

        test_only = False
        if test_only:
            '''
            confmat = evaluate(model_ft, data_loader_test, device=device, num_classes=num_classes)
            print(confmat)
            '''
            return

        ## parser_optimizer
        # Scale learning rate based on global batch size
        # cfg.INIT_LR = cfg.INIT_LR * float(self.batch_size_all) / 256
        optimizer_ft = parser_optimizer(cfg,model_ft)

        ## parser_lr_scheduler
        lr_scheduler_ft = parser_lr_scheduler(cfg, optimizer_ft)

        model_ft, optimizer_ft = amp.initialize(model_ft, optimizer_ft, opt_level=cfg.APEX_LEVEL)

        '''
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
        lf = lambda x: (((1 + math.cos(x * math.pi / self.cfg.N_MAX_EPOCHS)) / 2) ** 1.0) * 0.8 + 0.2  # cosine
        lr_scheduler_ft = lr_scheduler.LambdaLR(optimizer_ft, lr_lambda=lf)
        '''

        if cfg.distributed:
            model_ft = DistributedDataParallel(model_ft, delay_allreduce=True)

        '''
        # Freeze
        freeze = ['', ]  # parameter names to freeze (full or partial)
        if any(freeze):
            for k, v in model_ft.named_parameters():
                if any(x in k for x in freeze):
                    print('freezing %s' % k)
                    v.requires_grad = False
        '''

        if self.cfg.PRETRAIN_MODEL is not None:
            if self.cfg.RESUME:
                self.start_epoch = self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft, optimizer_ft, lr_scheduler_ft)
            else:
                self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft)

        ## vis net graph
        if self.cfg.TENSORBOARD_MODEL and False:
            self.tb_writer.add_graph(model_ft, (model_ft.dummy_input.cuda(),))

        self.n_steps_per_epoch = int(ceil(sum(len(t) for t in datasets['train'])))

        best_acc = 0.0
        for epoch in range(self.start_epoch + 1, self.cfg.N_MAX_EPOCHS):
            if cfg.distributed:
                dataloaders['train'].sampler.set_epoch(epoch)
                # data_samplers['train'].set_epoch(epoch)
            self.train_epoch(epoch, model_ft, dataloaders['train'], optimizer_ft, lr_scheduler_ft,None)
            lr_scheduler_ft.step()

            if self.cfg.DATASET.VAL:
                acc = self.val_epoch(epoch, model_ft, dataloaders['val'] , optimizer_ft, lr_scheduler_ft)

                if cfg.local_rank == 0:
                    # start to save best performance model after learning rate decay to 1e-6
                    if best_acc < acc:
                        self.ckpts.autosave_checkpoint(model_ft, epoch, 'best', optimizer_ft, lr_scheduler_ft,amp=None)
                        best_acc = acc
                        continue

            if not epoch % cfg.N_EPOCHS_TO_SAVE_MODEL:
                if cfg.local_rank == 0:
                    self.ckpts.autosave_checkpoint(model_ft, epoch,'autosave', optimizer_ft, lr_scheduler_ft,amp=None)

        self.tb_writer.close()

        dist.destroy_process_group() if cfg.local_rank!=0 else None
        torch.cuda.empty_cache()

    def train_epoch(self, epoch, model, dataloader, optimizer, lr_scheduler, grad_normalizer=None,prefix="train"):
        model.train()

        _timer = Timer()
        lossLogger = MetricLogger(delimiter="  ")
        performanceLogger = MetricLogger(delimiter="  ")

        # Creates once at the beginning of training
        scaler = torch.cuda.amp.GradScaler()

        for i, (imgs, targets) in enumerate(dataloader):
            _timer.tic()
            # zero the parameter gradients
            optimizer.zero_grad()

            # imgs = imgs.cuda()
            imgs = list(img.cuda() for img in imgs) if isinstance(imgs,list) else imgs.cuda()
            # labels = [label.cuda() for label in labels] if isinstance(labels,list) else labels.cuda()
            # labels = [{k: v.cuda() for k, v in t.items()} for t in labels] if isinstance(labels,list) else labels.cuda()
            if isinstance(targets,list):
                if isinstance(targets[0],torch.Tensor):
                    targets = [t.cuda() for t in targets]
                else:
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            else:
                targets = targets.cuda()

            out = model(imgs, targets, prefix)

            if not isinstance(out, tuple):
                losses, performances = out, None
            else:
                losses, performances = out

            self.n_iters_elapsed += 1

            with amp.scale_loss(losses["loss"], optimizer) as scaled_loss:
                scaled_loss.backward()

            optimizer.step()

            # torch.cuda.synchronize()
            _timer.toc()

            if (i + 1) % self.cfg.N_ITERS_TO_DISPLAY_STATUS == 0:
                print('losses',losses)
                if self.cfg.distributed:
                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_dict(losses)
                    print('loss_dict_reduced', loss_dict_reduced)
                    lossLogger.update(**loss_dict_reduced)
                else:
                    lossLogger.update(**losses)

                if performances is not None and all(performances):
                    if self.cfg.distributed:
                        # reduce performances over all GPUs for logging purposes
                        performance_dict_reduced = reduce_dict(performances)
                        performanceLogger.update(**performance_dict_reduced)
                    else:
                        performanceLogger.update(**performances)

                if self.cfg.local_rank == 0:
                    template = "[epoch {}/{}, iter {}, lr {}] Total train loss: {:.4f} " "(ips = {:.2f})\n" "{}"
                    logger.info(
                        template.format(
                            epoch, self.cfg.N_MAX_EPOCHS, i,
                            round(get_current_lr(optimizer), 6),
                            lossLogger.meters["loss"].value,
                            self.batch_size * self.cfg.N_ITERS_TO_DISPLAY_STATUS /  _timer.total_time,
                            "\n".join(["{}: {:.4f}".format(n, l.value) for n, l in lossLogger.meters.items() if n != "loss"]),
                        )
                    )

            del imgs, targets

        if self.cfg.TENSORBOARD and self.cfg.local_rank == 0:
            # Logging train losses
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l.global_avg, epoch) for n, l in lossLogger.meters.items()]
            if len(performanceLogger.meters):
                [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v.global_avg, epoch) for k, v in performanceLogger.meters.items()]

        if self.cfg.TENSORBOARD_WEIGHT and False:
            for name, param in model.named_parameters():
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                self.tb_writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    @torch.no_grad()
    def val_epoch(self, epoch, model, dataloader, optimizer=None, lr_scheduler=None,prefix="val"):
        model.eval()

        lossLogger = MetricLogger(delimiter="  ")
        performanceLogger = MetricLogger(delimiter="  ")

        with torch.no_grad():
            for (imgs, targets) in dataloader:

                # imgs = imgs.cuda()
                imgs = list(img.cuda() for img in imgs) if isinstance(imgs, list) else imgs.cuda()
                # labels = [label.cuda() for label in labels] if isinstance(labels,list) else labels.cuda()
                # labels = [{k: v.cuda() for k, v in t.items()} for t in labels] if isinstance(labels,list) else labels.cuda()
                if isinstance(targets, list):
                    if isinstance(targets[0], torch.Tensor):
                        targets = [t.cuda() for t in targets]
                    else:
                        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                else:
                    targets = targets.cuda()

                losses, performances = model(imgs, targets, prefix)

                if self.cfg.distributed:
                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_dict(losses)
                    lossLogger.update(**loss_dict_reduced)
                else:
                    lossLogger.update(**losses)

                if performances is not None and all(performances):
                    if self.cfg.distributed:
                        # reduce performances over all GPUs for logging purposes
                        performance_dict_reduced = reduce_dict(performances)
                        performanceLogger.update(**performance_dict_reduced)
                    else:
                        performanceLogger.update(**performances)

                del imgs, targets

        if self.cfg.TENSORBOARD and self.cfg.local_rank == 0:
            # Logging val Loss
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l.global_avg, epoch) for n, l in lossLogger.meters.items()]
            if len(performanceLogger.meters):
                # Logging val performances
                [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v.global_avg, epoch) for k, v in performanceLogger.meters.items()]

        if self.cfg.local_rank == 0:
            template = "[epoch {}] Total {} loss : {:.4f} " "\n" "{}"
            logger.info(
                template.format(
                    epoch,prefix,lossLogger.meters["loss"].global_avg,
                    "\n".join(["{}: {:.4f}".format(n, l.global_avg) for n, l in lossLogger.meters.items() if n != "loss"]),
                )
            )

            perf_log_str = f"\n------------ Performances ({prefix}) ----------\n"
            for k,v in performanceLogger.meters.items():
                perf_log_str += "{:}: {:.4f}\n".format(k, v.global_avg)
            perf_log_str += "------------------------------------\n"
            logger.info(perf_log_str)

        acc = performanceLogger.meters['performance'].global_avg

        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
    parser.add_argument('--setting', default='conf/hymenoptera.yml', help='The path to the configuration file.')

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    cfg = CommonConfiguration.from_yaml(args.setting)

    cfg.distributed = False
    if 'WORLD_SIZE' in os.environ:
        cfg.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(list(map(str, cfg.GPU_IDS)))

    cfg.gpu = cfg.GPU_IDS[0]
    cfg.world_size = 1
    cfg.local_rank = args.local_rank

    if cfg.distributed:
        cfg.gpu = cfg.local_rank
        torch.cuda.set_device(cfg.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        cfg.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    logger.info('Loaded configuration file: {}'.format(args.setting))
    logger.info('Use gpu ids: {}'.format(cfg.GPU_IDS))

    trainer = Trainer(cfg)
    logger.info('Begin to training ...')
    trainer.run()

    logger.info('finish!')
    torch.cuda.empty_cache()

    '''
    cfg.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    cfg.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    cfg.distributed = cfg.world_size > 1
    cfg.rank = args.local_rank

    print(cfg.rank,cfg.world_size,cfg.GPU_IDS,len(cfg.GPU_IDS))
    if cfg.rank== -1 and len(cfg.GPU_IDS)>0:
        cfg.GPU_IDS = [cfg.GPU_IDS[0]]
    assert (cfg.rank != -1 and len(cfg.GPU_IDS)>1) or (cfg.rank == -1 and len(cfg.GPU_IDS)==1) ,'--rank must be compatible with len(cfg.GPU_IDS)'

    # torch_distributed_zero_first(cfg.rank)

    # DDP mode
    if cfg.rank != -1:
        assert torch.cuda.device_count() > cfg.rank
        torch.cuda.set_device(cfg.rank)
        device = torch.device('cuda', cfg.rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
    else:
        device = torch.device('cuda:'+str(cfg.GPU_IDS[0]) if torch.cuda.is_available() else 'cpu')

    cfg.device = device

    if cfg.rank in [-1, 0]:
        logger.info('Loaded configuration file: {}'.format(args.setting))
        logger.info('Use gpu ids: {}'.format(cfg.GPU_IDS))

    trainer = Trainer(cfg)
    logger.info('Begin to training ...')
    trainer.run()

    if cfg.rank in [-1, 0]:
        logger.info('finish!')
        torch.cuda.empty_cache()
    '''