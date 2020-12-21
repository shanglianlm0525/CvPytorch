# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/10 16:24
# @Author : liumin
# @File : trainer.py

import argparse
import os
from collections import defaultdict

import math
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
from src.evaluation import build_evaluator
from src.utils.distributed import LossLogger
from src.optimizers import build_optimizer
from src.lr_schedulers import build_lr_scheduler
from src.utils.freeze import freeze_models

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


torch.set_default_tensor_type(torch.FloatTensor)

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


# logger = logging.getLogger("pytorch")

class Trainer:
    def __init__(self, cfg):
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

        dataloaders = {x: DataLoader(datasets[x], batch_size=cfg.DATASET[x.upper()].BATCH_SIZE, sampler=data_samplers[x],
                                     num_workers=cfg.DATASET[x.upper()].NUM_WORKER, collate_fn=dataset_class.collate_fn if hasattr(dataset_class,'collate_fn') else default_collate,
                          pin_memory=True, drop_last=True) for x in ['train', 'val']}  # collate_fn=detection_collate,

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

        ## parser_optimizer
        # Scale learning rate based on global batch size
        if cfg.SCALE_LR:
            cfg.INIT_LR = cfg.INIT_LR * float(self.batch_size) / 256
        optimizer_ft = build_optimizer(cfg,model_ft)

        ## parser_lr_scheduler
        lr_scheduler_ft = build_lr_scheduler(cfg, optimizer_ft)

        if cfg.distributed:
            model_ft = DDP(model_ft, device_ids=[cfg.local_rank], output_device=(cfg.local_rank))

        # Freeze
        freeze_models(model_ft)

        if self.cfg.PRETRAIN_MODEL is not None:
            if self.cfg.RESUME:
                self.start_epoch = self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft, optimizer_ft, lr_scheduler_ft)
            else:
                self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL,model_ft)

        ## vis net graph
        if self.cfg.TENSORBOARD_MODEL and False:
            self.tb_writer.add_graph(model_ft, (model_ft.dummy_input.cuda(),))

        self.steps_per_epoch = int(dataset_sizes['train']//self.batch_size)

        best_acc = 0.0
        scaler = amp.GradScaler(enabled=True)
        for epoch in range(self.start_epoch + 1, self.cfg.N_MAX_EPOCHS):
            if cfg.distributed:
                dataloaders['train'].sampler.set_epoch(epoch)
            self.train_epoch(scaler, epoch, model_ft,datasets['train'], dataloaders['train'], optimizer_ft)
            lr_scheduler_ft.step()

            if self.cfg.DATASET.VAL and (not epoch % cfg.EVALUATOR.EVAL_INTERVALS or epoch==self.cfg.N_MAX_EPOCHS-1):
                acc = self.val_epoch(epoch, model_ft,datasets['val'], dataloaders['val'])

                if cfg.local_rank == 0:
                    # start to save best performance model after learning rate decay to 1e-6
                    if best_acc < acc:
                        self.ckpts.autosave_checkpoint(model_ft, epoch, 'best', optimizer_ft, lr_scheduler_ft)
                        best_acc = acc
                        # continue

            if not epoch % cfg.N_EPOCHS_TO_SAVE_MODEL:
                if cfg.local_rank == 0:
                    self.ckpts.autosave_checkpoint(model_ft, epoch,'last', optimizer_ft, lr_scheduler_ft)

        if cfg.local_rank == 0:
            self.tb_writer.close()

        dist.destroy_process_group() if cfg.local_rank!=0 else None
        torch.cuda.empty_cache()

    def train_epoch(self, scaler, epoch, model, dataset, dataloader, optimizer, prefix="train"):
        model.train()

        _timer = Timer()
        lossLogger = LossLogger()
        performanceLogger = build_evaluator(self.cfg, dataset)

        for i, sample in enumerate(dataloader):
            imgs, targets = sample['image'], sample['target']
            _timer.tic()
            # zero the parameter gradients
            optimizer.zero_grad()

            imgs = list(img.cuda() for img in imgs) if isinstance(imgs, list) else imgs.cuda()
            if isinstance(targets, list):
                if isinstance(targets[0], torch.Tensor):
                    targets = [t.cuda() for t in targets]
                elif isinstance(targets[0], np.ndarray):
                    targets = [torch.from_numpy(t).cuda() for t in targets]
                else:
                    targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
            else:
                targets = targets.cuda()

            # Autocast
            with amp.autocast(enabled=True):
                out = model(imgs, targets, prefix)

            if not isinstance(out, tuple):
                losses, predicts = out, None
            else:
                losses, predicts = out

            self.n_iters_elapsed += 1

            # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
            # Backward passes under autocast are not recommended.
            # Backward ops run in the same dtype autocast chose for corresponding forward ops.
            scaler.scale(losses["loss"]).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)
            # Updates the scale for next iteration.
            scaler.update()

            # torch.cuda.synchronize()
            _timer.toc()

            if (i + 1) % self.cfg.N_ITERS_TO_DISPLAY_STATUS == 0:
                if self.cfg.distributed:
                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_dict(losses)
                    lossLogger.update(**loss_dict_reduced)
                    del loss_dict_reduced
                else:
                    lossLogger.update(**losses)

                if predicts is not None:
                    if self.cfg.distributed:
                        # reduce performances over all GPUs for logging purposes
                        predicts_dict_reduced = reduce_dict(predicts)
                        performanceLogger.update(targets, predicts_dict_reduced)
                        del predicts_dict_reduced
                    else:
                        performanceLogger.update(targets, predicts)
                    del predicts

                if self.cfg.local_rank == 0:
                    template = "[epoch {}/{}, iter {}, lr {}] Total train loss: {:.4f} " "(ips = {:.2f})\n" "{}"
                    logger.info(
                        template.format(
                            epoch, self.cfg.N_MAX_EPOCHS-1, i,
                            round(get_current_lr(optimizer), 6),
                            lossLogger.meters["loss"].value,
                            self.batch_size * self.cfg.N_ITERS_TO_DISPLAY_STATUS / _timer.diff,
                            "\n".join(
                                ["{}: {:.4f}".format(n, l.value) for n, l in lossLogger.meters.items() if n != "loss"]),
                        )
                    )

            del imgs, targets, losses

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
                imgs, targets = sample['image'], sample['target']
                imgs = list(img.cuda() for img in imgs) if isinstance(imgs, list) else imgs.cuda()
                if isinstance(targets, list):
                    if isinstance(targets[0], torch.Tensor):
                        targets = [t.cuda() for t in targets]
                    elif isinstance(targets[0], np.ndarray):
                        targets = [torch.from_numpy(t).cuda() for t in targets]
                    else:
                        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
                else:
                    targets = targets.cuda()

                losses, predicts = model(imgs, targets, prefix)

                if self.cfg.distributed:
                    # reduce losses over all GPUs for logging purposes
                    loss_dict_reduced = reduce_dict(losses)
                    lossLogger.update(**loss_dict_reduced)
                    del loss_dict_reduced
                else:
                    lossLogger.update(**losses)

                if predicts is not None:
                    if self.cfg.distributed:
                        # reduce performances over all GPUs for logging purposes
                        predicts_dict_reduced = reduce_dict(predicts)
                        performanceLogger.update(targets, predicts_dict_reduced)
                        del predicts_dict_reduced
                    else:
                        performanceLogger.update(targets, predicts)
                    del predicts

                del imgs, targets, losses

        performances = performanceLogger.evaluate()
        if self.cfg.TENSORBOARD and self.cfg.local_rank == 0:
            # Logging val Loss
            [self.tb_writer.add_scalar(f"loss/{prefix}_{n}", l.global_avg, epoch) for n, l in lossLogger.meters.items()]
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

            perf_log_str = f"\n------------ Performances ({prefix}) ----------\n"
            for k, v in performances.items():
                perf_log_str += "{:}: {:.4f}\n".format(k, v)
            perf_log_str += "------------------------------------\n"
            logger.info(perf_log_str)

        acc = performances['performance']

        return acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
    parser.add_argument('--setting', default='conf/hymenoptera.yml', help='The path to the configuration file.')

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