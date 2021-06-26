# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/17 9:50
# @Author : liumin
# @File : det_transforms.py
import collections
import copy
import functools
import random
import warnings
from collections import Sequence
from numbers import Number
import cv2
import numbers

import math
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode
from pycocotools import mask as coco_mask
import numpy as np

from src.models.fcos_augment import random_crop_resize
from src.transforms.det_color import color_aug_and_norm
from src.transforms.det_warp import warp_and_resize
from src.transforms.seg_transforms import _setup_size

__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Resize', 'RandomResizedCrop',
        'RandomRotate',
        'ColorJitter', 'GaussianBlur',
        'Normalize', 'ToTensor',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask',
        'FcosPreprocess']


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor(object):
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if self.normalize:
            return {'image': F.to_tensor(img), 'target': target}
        else:
            return {'image': torch.from_numpy( np.array(img, dtype=np.float32).transpose(2, 0, 1) ), 'target': target }

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        return {'image': F.normalize(img, self.mean, self.std),'target': target}


class Resize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, alignment=False, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.alignment = alignment
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img, target = sample['image'], sample['target']
        w, h = img.size
        boxes = target["boxes"]

        if self.alignment:
            min_side, max_side = self.size
            small_side = min(w, h)
            large_side = max(w, h)
            align_scale = min_side / small_side
            if large_side * align_scale > max_side:
                align_scale = max_side / large_side
            align_W, align_H = int(align_scale * w), int(align_scale * h)

            # pad_W = 32 - align_W % 32
            # pad_H = 32 - align_H % 32

            img_resized = F.resize(img, (align_W, align_H), self.interpolation)
            # img_paded = np.zeros(shape=[align_H + pad_H, align_W + pad_W, 3], dtype=np.uint8)
            # img_paded[:align_H, :align_W, :] = img_resized

            boxes[:, [0, 2]] = boxes[:, [0, 2]] * align_scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * align_scale

            target["scales"] = torch.tensor(align_scale, dtype=torch.float)
            target["boxes"] = boxes  # boxes.numpy()
            return {'image': img_resized, 'target': target}
        else:
            if isinstance(self.size, int):
                scale = min(self.size / h, self.size / w)
                boxes[:, 0::2] = boxes[:, 0::2] * scale
                boxes[:, 1::2] = boxes[:, 1::2] * scale
            else:
                scale_h, scale_w = self.size[0] / w , self.size[1] / h
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h

            target["boxes"] = boxes
            return {'image': F.resize(img, self.size, self.interpolation), 'target': target}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            w, h = img.size
            boxes = target["boxes"]
            if boxes.shape[0] != 0:
                xmin = w - boxes[:, 2]
                xmax = w - boxes[:, 0]
                boxes[:, 2] = xmax
                boxes[:, 0] = xmin
            target["boxes"] = boxes
            return {'image': F.hflip(img),'target': target}
        return {'image': img,'target': target}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            w, h = img.size
            boxes = target["boxes"]
            if boxes.shape[0] != 0:
                ymin = h - boxes[:, 3]
                ymax = h - boxes[:, 1]
                boxes[:, 3] = ymax
                boxes[:, 1] = ymin
            target["boxes"] = boxes
            return {'image': F.vflip(img), 'target': target}
        return {'image': img, 'target': target}


class ColorJitter(object):
    def __init__(self, p=0.3, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(img), 'target': target}
        return {'image': img, 'target': target}


class RandomRotate(object):
    def __init__(self, p=0.5, degrees=0):
        self.p = p
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            boxes = target["boxes"]
            angle = random.uniform(self.degrees[0], self.degrees[1])
            w, h = img.size
            rx0, ry0 = w / 2.0, h / 2.0
            img = img.rotate(angle)
            a = -angle / 180.0 * math.pi
            # boxes = torch.from_numpy(boxes)
            new_boxes = torch.zeros_like(boxes)
            new_boxes[:, 0] = boxes[:, 1]
            new_boxes[:, 1] = boxes[:, 0]
            new_boxes[:, 2] = boxes[:, 3]
            new_boxes[:, 3] = boxes[:, 2]

            for i in range(boxes.shape[0]):
                ymin, xmin, ymax, xmax = new_boxes[i, :]
                xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
                x0, y0 = xmin, ymin
                x1, y1 = xmin, ymax
                x2, y2 = xmax, ymin
                x3, y3 = xmax, ymax
                z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
                tp = torch.zeros_like(z)
                tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
                tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
                ymax, xmax = torch.max(tp, dim=0)[0]
                ymin, xmin = torch.min(tp, dim=0)[0]
                new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
            new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
            new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
            boxes[:, 0] = new_boxes[:, 1]
            boxes[:, 1] = new_boxes[:, 0]
            boxes[:, 2] = new_boxes[:, 3]
            boxes[:, 3] = new_boxes[:, 2]
            target["boxes"] = boxes # boxes.numpy()
        return {'image': img, 'target': target}


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), alignment=False, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")
            self.size = size

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.alignment = alignment

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        width, height = F._get_image_size(img)
        area = height * width

        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            log_ratio = torch.log(torch.tensor(ratio))
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        # boxes = torch.from_numpy(boxes)
        boxes -= torch.Tensor([i, j, i, j])
        w, h = img.size
        boxes = target["boxes"]
        boxes[:, 1::2].clamp_(min=0, max=h - 1)
        boxes[:, 0::2].clamp_(min=0, max=w - 1)

        if self.alignment:
            # crop
            img = F.crop(img, i, j, h, w)
            w, h = img.size
            min_side, max_side = self.size
            small_side = min(w, h)
            large_side = max(w, h)
            align_scale = min_side / small_side
            if large_side * align_scale > max_side:
                align_scale = max_side / large_side
            align_W, align_H = int(align_scale * w), int(align_scale * h)

            # pad_W = 32 - align_W % 32
            # pad_H = 32 - align_H % 32

            img_resized = F.resize(img, (align_W, align_H), self.interpolation)
            # print(img_resized.size)
            # img_paded = torch.zeros([align_H + pad_H, align_W + pad_W, 3], dtype=np.uint8)
            # img_paded[:align_H, :align_W, :] = img_resized

            boxes[:, [0, 2]] = boxes[:, [0, 2]] * align_scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * align_scale

            target["scales"] = torch.tensor(align_scale, dtype=torch.float)
            target["boxes"] = boxes  # boxes.numpy()
            return {'image': img_resized, 'target': target}
        else:
            if isinstance(self.size, int):
                scale = min(self.size / h, self.size / w)
                boxes[:, 0::2] = boxes[:, 0::2] * scale
                boxes[:, 1::2] = boxes[:, 1::2] * scale
            else:
                scale_h, scale_w = self.size[0] / w, self.size[1] / h
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h

            target["boxes"] = boxes # boxes.numpy()
            return {'image': F.resized_crop(img, i, j, h, w, self.size, self.interpolation), 'target': target}


class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur.
    If the image is torch Tensor, it is expected
    to have [..., C, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        kernel_size (int or sequence): Size of the Gaussian kernel.
        sigma (float or tuple of float (min, max)): Standard deviation to be used for
            creating kernel to perform blurring. If float, sigma is fixed. If it is tuple
            of float (min, max), sigma is chosen uniformly at random to lie in the
            given range.

    Returns:
        PIL Image or Tensor: Gaussian blurred version of the input image.

    """

    def __init__(self, p=0.5, kernel_size=5, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = _setup_size(kernel_size, "Kernel size should be a tuple/list of two integers")
        for ks in self.kernel_size:
            if ks <= 0 or ks % 2 == 0:
                raise ValueError("Kernel size value should be an odd and positive number.")

        if isinstance(sigma, numbers.Number):
            if sigma <= 0:
                raise ValueError("If sigma is a single number, it must be positive.")
            sigma = (sigma, sigma)
        elif isinstance(sigma, Sequence) and len(sigma) == 2:
            if not 0. < sigma[0] <= sigma[1]:
                raise ValueError("sigma values should be positive and of the form (min, max).")
        else:
            raise ValueError("sigma should be a single number or a list/tuple with length 2.")

        self.sigma = sigma

    @staticmethod
    def get_params(sigma_min: float, sigma_max: float) -> float:
        """Choose sigma for random gaussian blurring.

        Args:
            sigma_min (float): Minimum standard deviation that can be chosen for blurring kernel.
            sigma_max (float): Maximum standard deviation that can be chosen for blurring kernel.

        Returns:
            float: Standard deviation to be passed to calculate kernel for gaussian blurring.
        """
        return torch.empty(1).uniform_(sigma_min, sigma_max).item()


    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): image to be blurred.

        Returns:
            PIL Image or Tensor: Gaussian blurred image
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            sigma = self.get_params(self.sigma[0], self.sigma[1])
            return {'image': F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), 'target': target}
        return {'image': img, 'target': target}


def _box_inter(box1, box2):
    tl = torch.max(box1[:,None,:2], box2[:,:2])  # [n,m,2]
    br = torch.min(box1[:,None,2:], box2[:,2:])  # [n,m,2]
    hw = (br-tl).clamp(min=0)  # [n,m,2]
    inter = hw[:,:,0] * hw[:,:,1]  # [n,m]
    return inter


class FcosPreprocess(object):
    def __init__(self, p=0.5, size=[800,1333], scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), crop=False):
        self.p = p
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.crop = crop

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]

        if self.crop and random.random() < self.p:
            img, boxes = random_crop_resize(img, boxes)

        img = np.array(img)

        min_side, max_side = self.size
        h, w, _ = img.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(img, (nw, nh))
        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32
        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale

        target["scales"] = torch.tensor(scale, dtype=torch.float)
        target["boxes"] = boxes
        return {'image': Image.fromarray(image_paded), 'target': target}


class NanoDetPreprocess(object):
    def __init__(self,cfg, keep_ratio=True, dst_shape=[320, 320]):
        self.dst_shape = dst_shape
        self.warp = functools.partial(warp_and_resize,
                                      warp_kwargs=cfg,
                                      keep_ratio=keep_ratio)
        # self.color = functools.partial(color_aug_and_norm,kwargs=cfg)

    def __call__(self, sample):
        sample = self.warp(sample=sample, dst_shape=self.dst_shape)
        # sample = self.color(sample=sample)
        return sample


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            target["annotations"] = anno
            return {'image': img, 'target': target}
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return {'image': img, 'target': target}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        w, h = img.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["height"] = torch.tensor(h)
        target["width"] = torch.tensor(w)
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return {'image': img, 'target': target}
