# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/1/4 18:51
# @Author : liumin
# @File : trainer_fcos_coco_new.py


import argparse
import os
from collections import defaultdict

import math
import random
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

import attr
from tqdm import tqdm
from copy import deepcopy
from importlib import import_module
from time import time
from datetime import datetime
from math import ceil
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path as P
import torch.distributed as dist

from torch.cuda import amp
from torch.nn import SyncBatchNorm
from torch.nn.parallel import DistributedDataParallel as DDP

from src.utils.config import CommonConfiguration
from src.utils.logger import logger
from src.utils.timer import Timer
from src.utils.tensorboard import DummyWriter
from src.utils.checkpoints import Checkpoints
from src.utils.distributed import init_distributed, is_main_process, reduce_dict
from src.evaluator import build_evaluator
from src.utils.distributed import LossLogger
from src.optimizers import build_optimizer, get_current_lr
from src.lr_schedulers import build_lr_scheduler
from src.utils.freeze import freeze_models
from src.lr_schedulers.warmup import get_warmup_lr
from src.datasets.prefetch_dataLoader import PrefetchDataLoader

torch.backends.cudnn.enabled = True
torch.set_default_tensor_type(torch.FloatTensor)

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_class_name(full_class_name):
    return full_class_name.split(".")[-1]


# logger = logging.getLogger("pytorch")

class Trainer:
    def __init__(self, cfg):
        seed_torch(1029)
        self.cfg = cfg
        self.start_epoch = -1
        self.n_iters_elapsed = 0
        self.device = self.cfg.GPU_IDS
        self.batch_size = self.cfg.DATASET.TRAIN.BATCH_SIZE * len(self.cfg.GPU_IDS)

        self.n_steps_per_epoch = None
        if cfg.local_rank == 0:
            self.experiment_id = self.experiment_id(self.cfg)
            self.ckpts = Checkpoints(logger,self.cfg.CHECKPOINT_DIR,self.experiment_id)
            self.tb_writer = DummyWriter(log_dir="%s/%s" % (self.cfg.TENSORBOARD_LOG_DIR, self.experiment_id))

    def experiment_id(self, cfg):
        return f"{cfg.EXPERIMENT_NAME}#{cfg.USE_MODEL.split('.')[-1]}#{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    def _parser_dict(self):
        dictionary = CommonConfiguration.from_yaml(cfg.DATASET.DICTIONARY)
        return dictionary[cfg.DATASET.DICTIONARY_NAME]

    def _parser_datasets(self):
        *dataset_str_parts, dataset_class_str = cfg.DATASET.CLASS.split(".")
        dataset_class = getattr(import_module(".".join(dataset_str_parts)), dataset_class_str)

        datasets = {x: dataset_class(data_cfg=cfg.DATASET[x.upper()], dictionary=self.dictionary, transform=None,
                                     target_transform=None, stage=x) for x in ['train', 'val']}

        data_samplers = defaultdict()
        if self.cfg.distributed:
            data_samplers = {x: DistributedSampler(datasets[x],shuffle=cfg.DATASET[x.upper()].SHUFFLE) for x in ['train', 'val']}
        else:
            data_samplers['train'] = RandomSampler(datasets['train'])
            data_samplers['val'] = SequentialSampler(datasets['val'])

        dataloaders = {
            x: PrefetchDataLoader(datasets[x], batch_size=cfg.DATASET[x.upper()].BATCH_SIZE, sampler=data_samplers[x],
                          num_workers=cfg.DATASET[x.upper()].NUM_WORKER,
                          collate_fn=dataset_class.collate_fn if hasattr(dataset_class,
                                                                         'collate_fn') else default_collate,
                          pin_memory=True, drop_last=True) for x in ['train', 'val']}

        dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
        return datasets, dataloaders, data_samplers, dataset_sizes


    def _parser_model(self):
        *model_mod_str_parts, model_class_str = self.cfg.USE_MODEL.split(".")
        model_class = getattr(import_module(".".join(model_mod_str_parts)), model_class_str)
        model = model_class(dictionary=self.dictionary)

        if self.cfg.distributed:
            model = SyncBatchNorm.convert_sync_batchnorm(model).cuda()
        else:
            model = model.cuda()

        return model

    def clip_grad(self, model):
        if self.cfg.GRAD_CLIP.TYPE == "norm":
            clip_method = clip_grad_norm_
        elif self.cfg.GRAD_CLIP.TYPE == "value":
            clip_method = clip_grad_value_
        else:
            raise ValueError(
                f"Only support 'norm' and 'value' as the grad_clip type, but {self.cfg.GRAD_CLIP.TYPE} is given."
            )

        clip_method(model.parameters(), self.cfg.GRAD_CLIP.VALUE)

    def run_step(self, scaler, model, sample, optimizer, lossLogger, performanceLogger, prefix):
        '''
            Training step including forward
            :param model: model to train
            :param sample: a batch of input data
            :param optimizer:
            :param lossLogger:
            :param performanceLogger:
            :param prefix: train or val or infer
            :return: losses, predicts
        '''
        imgs, targets = sample['image'], sample['target']
        imgs = list(img.cuda() for img in imgs) if isinstance(imgs, list) else imgs.cuda()
        if isinstance(targets, list):
            if isinstance(targets[0], torch.Tensor):
                targets = [t.cuda() for t in targets]
            elif isinstance(targets[0], np.ndarray):
                targets = [torch.from_numpy(t).cuda() for t in targets]
            else:
                targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        elif isinstance(targets, dict):
            for (k, v) in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.cuda()
                elif isinstance(v, list):
                    if isinstance(v[0], torch.Tensor):
                        targets[k] = [t.cuda() for t in v]
                    elif isinstance(v[0], np.ndarray):
                        targets[k] = [torch.from_numpy(t).cuda() for t in v]
        else:
            targets = targets.cuda()

        if prefix=='train':
            # zero the parameter gradients
            optimizer.zero_grad()

            # Autocast
            with amp.autocast(enabled=True):
                out = model(imgs, targets, prefix)
                if not isinstance(out, tuple):
                    losses, predicts = out, None
                else:
                    losses, predicts = out

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(losses["loss"]).backward()

            if self.cfg.GRAD_CLIP and self.cfg.GRAD_CLIP.VALUE:
                self.clip_grad(model)

            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()
        else:
            losses, predicts = model(imgs, targets, prefix)

        if lossLogger is not None:
            if self.cfg.distributed:
                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = reduce_dict(losses)
                lossLogger.update(**loss_dict_reduced)
                del loss_dict_reduced
            else:
                lossLogger.update(**losses)

        if performanceLogger is not None:
            if predicts is not None:
                if self.cfg.distributed:
                    # reduce performances over all GPUs for logging purposes
                    predicts_dict_reduced = reduce_dict(predicts)
                    performanceLogger.update(targets, predicts_dict_reduced)
                    del predicts_dict_reduced
                else:
                    performanceLogger.update(targets, predicts)
                del predicts

        del imgs, targets
        return losses

    def warm_up(self, scaler, model, dataloader, cfg, prefix='train'):
        optimizer = build_optimizer(cfg, model)
        model.train()

        cur_iter = 0
        while cur_iter < cfg.WARMUP.ITERS:
            for i, sample in enumerate(dataloader):
                cur_iter += 1
                if cur_iter >= cfg.WARMUP.ITERS:
                    break
                lr = get_warmup_lr(cur_iter, cfg)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                losses = self.run_step(scaler, model, sample, optimizer, None, None, prefix)

                if self.cfg.local_rank == 0:
                    template = "[iter {}/{}, lr {}] Total train loss: {:.4f} \n" "{}"
                    logger.info(
                        template.format(
                            cur_iter, cfg.WARMUP.ITERS, round(get_current_lr(optimizer), 6),
                            losses["loss"].item(),
                            "\n".join(
                                ["{}: {:.4f}".format(n, l.item()) for n, l in losses.items() if n != "loss"]),
                        )
                    )
        del optimizer


    def run(self):
        ## init distributed
        self.cfg = init_distributed(self.cfg)
        cfg = self.cfg
        # cfg.print()

        ## parser_dict
        self.dictionary = self._parser_dict()

        ## parser_datasets
        datasets, dataloaders,data_samplers, dataset_sizes = self._parser_datasets()

        ## parser_model
        model_ft = self._parser_model()

        # Scale learning rate based on global batch size
        if cfg.SCALE_LR:
            cfg.INIT_LR = cfg.INIT_LR * float(self.batch_size) / cfg.SCALE_LR

        scaler = amp.GradScaler(enabled=True)
        if cfg.WARMUP.NAME is not None and cfg.WARMUP.ITERS:
            logger.info('Start warm-up ... ')
            self.warm_up(scaler, model_ft, dataloaders['train'], cfg)
            logger.info('finish warm-up!')

        ## parser_optimizer
        optimizer_ft = build_optimizer(cfg, model_ft)

        ## parser_lr_scheduler
        lr_scheduler_ft = build_lr_scheduler(cfg, optimizer_ft)

        if cfg.distributed:
            model_ft = DDP(model_ft, device_ids=[cfg.local_rank], output_device=(cfg.local_rank))

        # Freeze
        freeze_models(model_ft)

        if self.cfg.PRETRAIN_MODEL is not None:
            if self.cfg.RESUME:
                self.start_epoch = self.ckpts.resume_checkpoint(model_ft, optimizer_ft)
            else:
                self.start_epoch = self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL, model_ft, optimizer_ft)

        ## vis network graph
        if self.cfg.TENSORBOARD_MODEL and False:
            self.tb_writer.add_graph(model_ft, (model_ft.dummy_input.cuda(),))

        self.steps_per_epoch = int(dataset_sizes['train']//self.batch_size)

        best_acc = 0.0
        best_perf_rst = None
        for epoch in range(self.start_epoch + 1, self.cfg.N_MAX_EPOCHS):
            if cfg.distributed:
                dataloaders['train'].sampler.set_epoch(epoch)
            self.train_epoch(scaler, epoch, model_ft,datasets['train'], dataloaders['train'], optimizer_ft)
            lr_scheduler_ft.step()

            if self.cfg.DATASET.VAL and (not epoch % cfg.EVALUATOR.EVAL_INTERVALS or epoch==self.cfg.N_MAX_EPOCHS-1):
                acc, perf_rst = self.val_epoch(epoch, model_ft,datasets['val'], dataloaders['val'])

                if cfg.local_rank == 0:
                    # start to save best performance model after learning rate decay to 1e-6
                    if best_acc < acc:
                        self.ckpts.autosave_checkpoint(model_ft, epoch, 'best', optimizer_ft)
                        best_acc = acc
                        best_perf_rst = perf_rst
                        # continue

            if not epoch % cfg.N_EPOCHS_TO_SAVE_MODEL:
                if cfg.local_rank == 0:
                    self.ckpts.autosave_checkpoint(model_ft, epoch,'last', optimizer_ft)

        if best_perf_rst is not None:
            logger.info(best_perf_rst.replace("(val)", "(best)"))

        if cfg.local_rank == 0:
            self.tb_writer.close()

        dist.destroy_process_group() if cfg.local_rank!=0 else None
        torch.cuda.empty_cache()

    def train_epoch(self, scaler, epoch, model, dataset, dataloader, optimizer, prefix="train"):
        model.train()

        _timer = Timer()
        lossLogger = LossLogger()
        performanceLogger = build_evaluator(self.cfg, dataset)

        num_iters = len(dataloader)
        for i, sample in enumerate(dataloader):
            self.n_iters_elapsed += 1
            _timer.tic()
            self.run_step(scaler, model, sample, optimizer, lossLogger, performanceLogger, prefix)
            torch.cuda.synchronize()
            _timer.toc()

            if (i + 1) % self.cfg.N_ITERS_TO_DISPLAY_STATUS == 0:
                if self.cfg.local_rank == 0:
                    template = "[epoch {}/{}, iter {}/{}, lr {}] Total train loss: {:.4f} " "(ips = {:.2f})\n" "{}"
                    logger.info(
                        template.format(
                            epoch, self.cfg.N_MAX_EPOCHS - 1, i, num_iters - 1,
                            round(get_current_lr(optimizer), 6),
                            lossLogger.meters["loss"].value,
                                   self.batch_size * self.cfg.N_ITERS_TO_DISPLAY_STATUS / _timer.diff,
                            "\n".join(
                                ["{}: {:.4f}".format(n, l.value) for n, l in lossLogger.meters.items() if n != "loss"]),
                        )
                    )

        if self.cfg.TENSORBOARD and self.cfg.local_rank == 0:
            # Logging train losses
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l.global_avg, epoch) for n, l in lossLogger.meters.items()]
            performances = performanceLogger.evaluate()
            if performances is not None and len(performances):
                [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v, epoch) for k, v in performances.items()]

        if self.cfg.TENSORBOARD_WEIGHT and False:
            for name, param in model.named_parameters():
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                self.tb_writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    @torch.no_grad()
    def val_epoch(self, epoch, model, dataset, dataloader, prefix="val"):
        model.eval()

        lossLogger = LossLogger()
        performanceLogger = build_evaluator(self.cfg, dataset)

        with torch.no_grad():
            for sample in dataloader:
                self.run_step(None, model, sample, None, lossLogger, performanceLogger, prefix)

        if self.cfg.TENSORBOARD and self.cfg.local_rank == 0:
            # Logging val Loss
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l.global_avg, epoch) for n, l in lossLogger.meters.items()]
            performances = performanceLogger.evaluate()
            if performances is not None and len(performances):
                # Logging val performances
                [self.tb_writer.add_scalar(f"performance/{prefix}_{k}", v, epoch) for k, v in performances.items()]

        if self.cfg.local_rank == 0:
            template = "[epoch {}] Total {} loss : {:.4f} " "\n" "{}"
            logger.info(
                template.format(
                    epoch, prefix, lossLogger.meters["loss"].global_avg,
                    "\n".join(
                        ["{}: {:.4f}".format(n, l.global_avg) for n, l in lossLogger.meters.items() if n != "loss"]),
                )
            )

            perf_log = f"\n------------ Performances ({prefix}) ----------\n"
            for k, v in performances.items():
                perf_log += "{:}: {:.4f}\n".format(k, v)
            perf_log += "------------------------------------\n"
            logger.info(perf_log)

        acc = performances['performance']

        return acc, perf_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
    parser.add_argument('--setting', default='conf/cityscapes_pspnet.yml', help='The path to the configuration file.')

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    cfg = CommonConfiguration.from_yaml(args.setting)
    cfg.local_rank = args.local_rank

    if cfg.local_rank==0:
        logger.info('Loaded configuration file: {}'.format(args.setting))
        logger.info('Use gpu ids: {}'.format(cfg.GPU_IDS))

    trainer = Trainer(cfg)
    logger.info('Begin to training ...')
    trainer.run()

    if cfg.local_rank == 0:
        logger.info('finish!')
    torch.cuda.empty_cache()