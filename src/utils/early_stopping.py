# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/13 14:05
# @Author : liumin
# @File : early_stopping.py


class EarlyStopping:
    # YOLOv5 simple early stopper
    '''
        stopper = EarlyStopping(patience=30)

        # Stop Single-GPU
        if RANK == -1 and stopper(epoch=epoch, fitness=fi):
            break
    '''
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            print(f'EarlyStopping patience {self.patience} exceeded, stopping training.')
        return stop
