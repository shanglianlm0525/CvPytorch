# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/10 15:45
# @Author : liumin
# @File : train.py

import argparse
import torch
from trainer import Trainer
from src.utils.config import CommonConfiguration
from src.utils.logger import logger

parser = argparse.ArgumentParser(description='Generic Pytorch-based Training Framework')
# parser.add_argument('--training-setting', required=True, help='The path to the training setting file you want to use.')
parser.add_argument('--setting',default='conf/flower.yml', help='The path to the training setting file you want to use.')

def main(args):
    logger.info('Using torch {}'.format(torch.__version__))
    cfg = CommonConfiguration.from_yaml(args.setting)
    logger.info('Loaded training setting file: {}'.format(args.setting))

    trainer = Trainer(cfg)
    logger.info('Started training...')
    trainer.start()
    logger.info('Finished training.')

if __name__ == '__main__':
    main(parser.parse_args())