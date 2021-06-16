# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 9:15
# @Author : liumin
# @File : checkpoints1.py

import os
import torch

from src.utils.distributed import is_main_process


class Checkpoints():
    def __init__(self, logger=None, checkpoint_dir=None, experiment_id=None):
        self.logger = logger
        super(Checkpoints, self).__init__()
        self.checkpoint_dir = checkpoint_dir
        self.experiment_id = experiment_id
        self.checkpoint_path = self.get_checkpoint()

    def get_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, self.experiment_id)
        # create checkpoint folder to save model
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, '{type}.pth')
        return checkpoint_path


    def load_checkpoint(self, save_path, model, optimizer=None):
        """Loads the checkpoint from the given file."""
        error_str = "Checkpoint '{}' not found"
        assert os.path.exists(save_path), error_str.format(save_path)
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint_state = torch.load(save_path, map_location="cpu")
        model.load_state_dict(checkpoint_state["model"], strict=False)
        # Load the optimizer state (commonly not done when fine-tuning)
        if optimizer:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        self.logger.info("Checkpoint loaded from {}".format(save_path))
        return checkpoint_state["epoch"]

    def resume_checkpoint(self, model, optimizer):
        checkpoint_path = self.checkpoint_path.format('last')
        chkpnt = torch.load(checkpoint_path)
        model.load_state_dict(chkpnt["model"], strict=False)
        start_epoch = chkpnt["epoch"]
        optimizer.load_state_dict(chkpnt["optimizer"])
        self.logger.info("Resumed checkpoint: {}".format('xxx'))
        return start_epoch


    def save_checkpoint(self, model, checkpoint_path, epoch=-1,type='last', optimizer=None):
        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(checkpoint_state, checkpoint_path)
        if type=='best':
            torch.save(model.state_dict(), checkpoint_path.replace('best','deploy'))
        self.logger.info("Checkpoint saved to {}".format(checkpoint_path))

    def autosave_checkpoint(self, model, epoch, type, optimizer):
        if is_main_process():
            checkpoint_path = self.checkpoint_path.format(type=type)
            self.save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                type = type,
                model=model,
                optimizer=optimizer
            )
