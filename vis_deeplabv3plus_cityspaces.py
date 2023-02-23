# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/6/23 10:26
# @Author : liumin
# @File : vis_deeplabv3plus_cityspaces.py
import argparse
from collections import defaultdict
from importlib import import_module

import torch
from datetime import datetime
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler
from torch.utils.data.dataloader import default_collate
from src.data.datasets import PrefetchDataLoader
from src.data.transforms import build_transforms
from src.utils.checkpoints import Checkpoints
from src.utils.config import CommonConfiguration
from src.utils.distributed import init_distributed
from src.utils.global_logger import logger



class Infer:
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

    def experiment_id(self, cfg):
        return f"{cfg.EXPERIMENT_NAME}#{cfg.USE_MODEL.split('.')[-1]}#{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"

    def _parser_dict(self):
        dictionary = CommonConfiguration.from_yaml(cfg.DATASET.DICTIONARY)
        return dictionary[cfg.DATASET.DICTIONARY_NAME]

    def _parser_transform(self, mode):
        return build_transforms(cfg.DATASET.DICTIONARY_NAME,cfg.DATASET[mode.upper()].TRANSFORMS,mode)

    def _parser_datasets(self, stage=['val']):
        *dataset_str_parts, dataset_class_str = cfg.DATASET.CLASS.split(".")
        dataset_class = getattr(import_module(".".join(dataset_str_parts)), dataset_class_str)

        datasets = {x: dataset_class(data_cfg=cfg.DATASET[x.upper()], dictionary=self.dictionary,
                                     transform=self._parser_transform(x),
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
        model = model.cuda()
        return model

    def run(self):
        ## init distributed
        self.cfg = init_distributed(self.cfg)
        cfg = self.cfg
        # cfg.print()

        ## parser_dict
        self.dictionary = self._parser_dict()

        stage = ['val']
        ## parser_datasets
        datasets, dataloaders,data_samplers, dataset_sizes = self._parser_datasets(stage)

        ## parser_model
        model_ft = self._parser_model()

        if self.cfg.PRETRAIN_MODEL is not None:
            self.start_epoch = self.ckpts.load_checkpoint(self.cfg.PRETRAIN_MODEL, model_ft)
        else:
            raise FileNotFoundError(' {} is not found!'.format(self.cfg.PRETRAIN_MODEL))


        self.steps_per_epoch = int(dataset_sizes['train']//self.batch_size)

        acc, perf_rst = self.val_epoch(0, model_ft,datasets['val'], dataloaders['val'])
        if perf_rst is not None:
            logger.info(perf_rst.replace("(val)", "(best)"))

        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
    # parser.add_argument('--setting', default='conf/cityscapes_deeplabv3plus.yml', help='The path to the configuration file.')
    parser.add_argument('--setting', default='conf/voc_deeplabv3plus.yml', help='The path to the configuration file.')
    # parser.add_argument('--setting', default='conf/camvid_enet.yml', help='The path to the configuration file.')
    # parser.add_argument('--setting', default='conf/coco_maskrcnn.yml', help='The path to the configuration file.')
    # parser.add_argument('--setting', default='conf/pennfudan_maskrcnn.yml', help='The path to the configuration file.')
    # parser.add_argument('--setting', default='conf/coco_yolov5_old.yml', help='The path to the configuration file.')

    # distributed training parameters
    parser.add_argument("--local_rank", default=0, type=int)

    args = parser.parse_args()
    cfg = CommonConfiguration.from_yaml(args.setting)
    cfg.local_rank = args.local_rank

    if cfg.local_rank == 0:
        logger.info('Loaded configuration file: {}'.format(args.setting))
        logger.info('Use gpu ids: {}'.format(cfg.GPU_IDS))

    infer = Infer(cfg)
    logger.info('Begin to training ...')
    infer.run()

    if cfg.local_rank == 0:
        logger.info('finish!')
    torch.cuda.empty_cache()