# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/10 15:46
# @Author : liumin
# @File : infer.py

import argparse
import importlib
import os

import cv2
import torch
import torchvision
import numpy as np
from itertools import zip_longest

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as transformsT

from tqdm import tqdm
from pathlib import Path as P

from src.utils.logger import logger
from src.utils.config import CommonConfiguration
from src.utils.checkpoints import Checkpoints, load_checkpoint


cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 80, 100,
    0, 0, 230,
    119, 11, 32,
]

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

parser = argparse.ArgumentParser(description="Generic Pytorch-based Training Framework")
parser.add_argument('--setting',default='conf/cityscapes_test.yml', help='The path to the training setting file you want to use.')
parser.add_argument("--model-path", default='checkpoints/Cityscapes_FastSCNN#FastSCNN#RMSprop#MultiStepLR#2020_08_04_16_25_02/Cityscapes_FastSCNN#FastSCNN#RMSprop#MultiStepLR#2020_08_04_16_25_02#best.pth', help='The storage location of the trained model.')
parser.add_argument("--indices", default=None, help='The indices of the test images.')
parser.add_argument("--dataset-dir", help='The path to the test images.')
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--save-dir", type=str, required=False)
parser.add_argument("--imgs-savedir", default='results', help="")
args = parser.parse_args()


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

        'infer': transformsT.Compose([
            transformsT.Resize((800,600)),
            transformsT.ToTensor(),
            transformsT.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    }
    return data_transforms

def eval(args):
    cfg = CommonConfiguration.from_yaml(args.setting)

    ## parser_dict
    dictionary = CommonConfiguration.from_yaml(cfg.DATASET.DICTIONARY)
    dictionary = next(dictionary.items())[1]

    prefix = 'infer'

    ## parser_datasets
    transforms = prepare_transforms_seg()
    *dataset_str_parts, dataset_class_str = cfg.DATASET.CLASS.split(".")
    dataset_class = getattr(importlib.import_module(".".join(dataset_str_parts)), dataset_class_str)

    dataset = dataset_class(data_cfg=cfg.DATASET[prefix.upper()],dictionary=dictionary, transform=transforms[prefix], stage=prefix)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=cfg.NUM_WORKERS,
                                 shuffle=False,  pin_memory=True) # collate_fn=detection_collate,

    ## parser model
    *model_mod_str_parts, model_class_str = cfg.USE_MODEL.split(".")
    model_class = getattr(importlib.import_module(".".join(model_mod_str_parts)), model_class_str)
    model_ft = model_class(dictionary=dictionary)


    load_checkpoint(args.model_path, model_ft)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
        model_ft.eval()

    for (imgs, imageids) in tqdm(dataloader):
        if cfg.HALF:
            imgs = imgs.half()

        with torch.no_grad():
            imgs = imgs.cuda()
            masks = model_ft(imgs, None, prefix)

            for mask,imageid in zip(masks,imageids):
                out_img = Image.fromarray(mask.astype(np.uint8))
                out_img.putpalette(cityspallete)

                outname = os.path.splitext(os.path.split(imageid)[-1])[0] + '.png'
                out_img.save(os.path.join(args.imgs_savedir, outname))

        '''
        rst_text = results["text"] if "text" in results.keys() else []
        rst_image = results["image"] if "image" in results.keys() else []

        if cfg.save_dir is not None:
            for im_path, res_im in zip_longest(im_paths, result_images):
                if results_save_dir is not None and res_im is not None:
                    if args.flatten_output_images:
                        _fname = str(P(im_path).relative_to(dataset_dir)).replace('/', '#')
                        if args.prepend_order_number_to_out_image_names:
                            _fname = f"{i:>06}#{_fname}"
                    else:
                        _fname = P(im_path).name

                    cv2.imwrite(str(dataset_res_save_dir / _fname), res_im)
        to_text_file(result_texts, dataset_dir / (res_save_name + ".txt"), mode="a")
        if results_save_dir is not None:
            to_text_file(result_texts, dataset_res_save_dir / (res_save_name + ".txt"), mode="a")

        print(f"Text file has been saved to "
              f"{dataset_dir / (res_save_name + '.txt')} and "
              f"{dataset_res_save_dir / (res_save_name + '.txt')}")
        '''



if __name__ == '__main__':
    eval(parser.parse_args())