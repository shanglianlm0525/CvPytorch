# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/12 9:15
# @Author : liumin
# @File : checkpoints.py

import os
import torch

from src.utils.distributed import is_main_process

# Common prefix for checkpoint file names
_NAME_PREFIX = "model_epoch_"
# Checkpoints directory name
_DIR_NAME = "checkpoints"


def load_checkpoint(save_path, model, optimizer=None, lr_scheduler=None,amp=None):
    """Loads the checkpoint from the given file."""
    error_str = "Checkpoint '{}' not found"
    assert os.path.exists(save_path), error_str.format(save_path)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint_state = torch.load(save_path, map_location="cpu")
    model.load_state_dict(checkpoint_state["model"], strict=False)
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint_state["optimizer"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler"])
    if amp:
        amp.load_state_dict(checkpoint_state["amp"])

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


    def load_checkpoint(self,save_path, model, optimizer=None,lr_scheduler=None):
        """Loads the checkpoint from the given file."""
        error_str = "Checkpoint '{}' not found"
        assert os.path.exists(save_path), error_str.format(save_path)
        # Load the checkpoint on CPU to avoid GPU mem spike
        checkpoint_state = torch.load(save_path, map_location="cpu")
        model.load_state_dict(checkpoint_state["model"], strict=False)
        # Load the optimizer state (commonly not done when fine-tuning)
        if optimizer:
            optimizer.load_state_dict(checkpoint_state["optimizer"])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler"])
        self.logger.info("Checkpoint loaded from {}".format(save_path))
        return checkpoint_state["epoch"]


    def get_last_checkpoint(self,checkpoint_dir):
        """Retrieves the most recent checkpoint (highest epoch number)."""
        # Checkpoint file names are in lexicographic order
        checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
        last_checkpoint_name = sorted(checkpoints)[-1]
        return os.path.join(checkpoint_dir, last_checkpoint_name)


    def resume_checkpoint(self, model, optimizer, lr_scheduler,amp):
        chkpnt = torch.load("xxx")
        model.load_state_dict(chkpnt["model"], strict=False)
        start_epoch = chkpnt["epoch"]
        optimizer.load_state_dict(chkpnt["optimizer"])
        lr_scheduler.load_state_dict(chkpnt["lr_scheduler"])
        amp.load_state_dict(chkpnt["amp"])
        self.logger.info("Resumed checkpoint: {}".format('xxx'))
        return start_epoch


    def save_checkpoint(self, model, checkpoint_path, epoch=-1,type='last', optimizer=None, lr_scheduler=None):
        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            # "lr_scheduler": lr_scheduler.state_dict(),
        }

        torch.save(checkpoint_state, checkpoint_path)
        if type=='best':
            torch.save(model.state_dict(), checkpoint_path.replace('best','deploy'))
        self.logger.info("Checkpoint saved to {}".format(checkpoint_path))

    def autosave_checkpoint(self,model, epoch, type, optimizer, lr_scheduler):
        if is_main_process():
            checkpoint_path = self.checkpoint_path.format(type='last')
            self.save_checkpoint(
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                type=type,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
            )
            if type=='best':
                checkpoint_path = self.checkpoint_path.format(type=type)
                self.save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    type = type,
                    model=model,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                )





'''
def save_cp(model, num, unit):
    save_path = self.save_dir / f"{self.train_id}#{cur_date}#{num}{unit}s.pth"
    save_checkpoint(model, save_path)


def autosave_cp(model, epoch, optimizer, lr_scheduler):
    save_checkpoint(
        model,
        self.get_autosave_path(),
        last_epoch=epoch,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

def resume_cp(self, model, optimizer, lr_scheduler):
    chkpnt = torch.load(self.get_autosave_path())
    model.load_state_dict(chkpnt["integrated_model"], strict=False)
    start_epoch = chkpnt["last_epoch"]
    n_iters_elapsed = chkpnt["n_iters_elapsed"]
    optimizer.load_state_dict(chkpnt["optimizer"])
    lr_scheduler.load_state_dict(chkpnt["lr_scheduler"])
    logger.info(f"Resumed checkpoint: {self.get_autosave_path()}")
    return start_epoch, n_iters_elapsed


def load_cp(self, model_path, model):
    if P(model_path).exists():
        _path = model_path
    elif (self.save_dir / model_path).exists():
        _path = str(self.save_dir / model_path)
    else:
        raise ValueError(f"Pretrain model {model_path} not found. Exit.")

    pretrain_model = torch.load(_path)
    model.load_state_dict(pretrain_model["integrated_model"], strict=False)
    logger.info(f"Loaded pretrain model: {_path}")

'''







'''
def get_checkpoint_dir():
    """Retrieves the location for storing checkpoints."""
    return os.path.join(cfg.OUT_DIR, _DIR_NAME)


def get_checkpoint(epoch):
    """Retrieves the path to a checkpoint file."""
    name = "{}{:04d}.pyth".format(_NAME_PREFIX, epoch)
    return os.path.join(get_checkpoint_dir(), name)


def get_last_checkpoint():
    """Retrieves the most recent checkpoint (highest epoch number)."""
    checkpoint_dir = get_checkpoint_dir()
    # Checkpoint file names are in lexicographic order
    checkpoints = [f for f in os.listdir(checkpoint_dir) if _NAME_PREFIX in f]
    last_checkpoint_name = sorted(checkpoints)[-1]
    return os.path.join(checkpoint_dir, last_checkpoint_name)


def has_checkpoint():
    """Determines if there are checkpoints available."""
    checkpoint_dir = get_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return False
    return any(_NAME_PREFIX in f for f in os.listdir(checkpoint_dir))


def save_checkpoint(model, optimizer, epoch):
    """Saves a checkpoint."""
    # Save checkpoints only from the master process
    if not dist.is_master_proc():
        return
    # Ensure that the checkpoint dir exists
    os.makedirs(get_checkpoint_dir(), exist_ok=True)
    # Omit the DDP wrapper in the multi-gpu setting
    sd = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
    # Record the state
    checkpoint = {
        "epoch": epoch,
        "model_state": sd,
        "optimizer_state": optimizer.state_dict(),
        "cfg": cfg.dump(),
    }
    # Write the checkpoint
    checkpoint_file = get_checkpoint(epoch + 1)
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def load_checkpoint(checkpoint_file, model, optimizer=None):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    # Load the checkpoint on CPU to avoid GPU mem spike
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model.module if cfg.NUM_GPUS > 1 else model
    ms.load_state_dict(checkpoint["model_state"])
    # Load the optimizer state (commonly not done when fine-tuning)
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint["epoch"]
'''
