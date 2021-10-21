# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/2 16:15
# @Author : liumin
# @File : det_transforms.py
import copy
import math
import random
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from pycocotools import mask as coco_mask


__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Resize', 'RandomResizedCrop', 'RandomCrop',
        'RandomAffine', 'RandomGrayscale', 'RandomRotation',
        'ColorHSV', 'ColorJitter', 'RandomEqualize', 'GaussianBlur', 'MedianBlur',
        'ToXYXY', 'ToCXCYWH', 'ToPercentCoords',
        'Cutout', 'Mosaic', 'CopyPaste',
        'Normalize', 'ToTensor', 'ToArrayImage',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']


def remove_small_boxes(boxes, min_size):
    """
       Remove boxes which contains at least one side smaller than min_size.

       Args:
           boxes (Tensor[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
               with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
           min_size (float): minimum size

       Returns:
           Tensor[K]: indices of the boxes that have both sides
           larger than min_size
       """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = np.where(keep)[0]
    return keep


def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (array[N, 4]): boxes in ``(x1, y1, x2, y2)`` format
            with ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        size (Tuple or List[height, width]): size of the image

    Returns:
        array[N, 4]: clipped boxes
    """
    height, width = size
    boxes[..., 0::2] = boxes[..., 0::2].clip(min=0, max=width)
    boxes[..., 1::2] = boxes[..., 1::2].clip(min=0, max=height)
    return boxes


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
        img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        target["boxes"] = torch.from_numpy(target["boxes"])
        if target.__contains__("masks"):
            mask = target["masks"]
            mask = mask.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            mask = np.ascontiguousarray(mask)
            target["masks"] = torch.from_numpy(mask)
        if self.normalize:
            return {'image': torch.from_numpy(img.astype(np.float32)).div_(255.0), 'target': target}
        else:
            return {'image': torch.from_numpy(img.astype(np.float32)), 'target': target }


class Normalize(object):
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        return {'image': F.normalize(img, self.mean, self.std),'target': target}


class ToArrayImage(object):
    def __call__(self, sample):
        if isinstance(sample, list):
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
            return sample
        else:
            img, target = sample['image'], sample['target']
            return {'image': np.asarray(img, dtype=np.uint8), 'target': target}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            height, width, _ = img.shape
            boxes = target["boxes"]
            if boxes.shape[0] != 0:
                xmin = width - 1 - boxes[:, 2]
                xmax = width - 1 - boxes[:, 0]
                boxes[:, 2] = xmax
                boxes[:, 0] = xmin
            target["boxes"] = boxes
            return {'image': cv2.flip(img, 1),'target': target}
        return {'image': img,'target': target}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            height, width, _ = img.shape
            boxes = target["boxes"]
            if boxes.shape[0] != 0:
                ymin = height -1 - boxes[:, 3]
                ymax = height -1 - boxes[:, 1]
                boxes[:, 3] = ymax
                boxes[:, 1] = ymin
            target["boxes"] = boxes
            return {'image': cv2.flip(img, 0), 'target': target}
        return {'image': img, 'target': target}



class Resize(object):
    """Resize the input img to the given size."""

    def __init__(self, size, keep_ratio=True, scaleup=True, fill=[128, 128, 128]):
        self.size = size if isinstance(size, list) else [size, size]
        self.keep_ratio = keep_ratio
        self.scaleup = scaleup # only valid when the keep_ratio is True
        self.fill = fill

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        height, width, _ = img.shape
        boxes = target["boxes"]

        if self.keep_ratio:
            scale = min(self.size[0] / height, self.size[1] / width)
            if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
                scale = min(scale, 1.0)
            oh, ow = int(round(height * scale)), int(round(width * scale))
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides

            if (height != oh) or (width != ow):
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border

            boxes[:, 1::2] = boxes[:, 1::2] * scale + top
            boxes[:, 0::2] = boxes[:, 0::2] * scale + left

            target["boxes"] = boxes
            target["pads"] = torch.tensor([top, left], dtype=torch.float)
            target["scales"] = torch.tensor([scale, scale], dtype=torch.float)
            return {'image': img, 'target': target}
        else:
            scale_h, scale_w = self.size[0] / height, self.size[1] / width
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w

            target["boxes"] = boxes
            target["pads"] = torch.tensor([0, 0], dtype=torch.float)
            target["scales"] = torch.tensor([scale_h, scale_w], dtype=torch.float)
            return {'image': img, 'target': target}


class RandomCrop(object):
    """Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.

            .. note::
                In torchscript mode padding as single int is not supported, use a sequence of
                length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image.
              If input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
              will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge.
              For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
              will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """
    def __init__(self, size, padding=None, pad_if_needed=False, fill=[128, 128, 128], min_size=3):
        super().__init__()
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.min_size = min_size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w, _ = img.shape
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw


    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        labels = target["labels"]
        if self.padding is not None:
            top, bottom, left, right = self.padding
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)
            boxes += np.array([left, top, left, top])

        height, width, _ = img.shape
        # pad the width if needed
        if self.pad_if_needed and (height < self.size[0] or width < self.size[1]):
            padh = self.size[0] - height
            pwdw = self.size[1] - width
            padh2, pwdw2 = padh // 2, pwdw // 2
            left, top, right, bottom = padh2, pwdw2, pwdw - padh2, padh - pwdw2
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border
            boxes += np.array([left, top, left, top])

        i, j, h, w = self.get_params(img, self.size)
        img = img[i:(i + h), j:(j + w), :]
        boxes -= np.array([j, i, j, i])

        # clip
        boxes = clip_boxes_to_image(boxes, [h, w])
        keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
        if keep.shape[0] < 1:
            img, target = sample['image'], sample['target']
            boxes = target["boxes"]
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            scale_h, scale_w = self.size[0] / h, self.size[1] / w
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            target["boxes"] = boxes
            return {'image': img, 'target': target}

        target["labels"] = labels[keep]
        target["boxes"] = boxes[keep]
        return {'image': img, 'target': target}


class RandomResizedCrop(object):
    """Crop a random portion of image and resize it to a given size.

    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of the original image is made: the crop has a random area (H * W)
    and a random aspect ratio. This crop is finally resized to the given
    size. This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of the crop, for each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

            .. note::
                In torchscript mode size as single int is not supported, use a sequence of length 1: ``[size, ]``.
        scale (tuple of float): Specifies the lower and upper bounds for the random area of the crop,
            before resizing. The scale is defined with respect to the area of the original image.
        ratio (tuple of float): lower and upper bounds for the random aspect ratio of the crop, before
            resizing.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` and
            ``InterpolationMode.BICUBIC`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.

    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), keep_ratio=True, fill=[128, 128, 128], min_size = 3):
        super().__init__()
        self.size = size
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.scale = scale
        self.ratio = ratio
        self.keep_ratio = keep_ratio
        self.fill = fill
        self.min_size = min_size

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
        height, width, _ = img.shape
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
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
        labels = target["labels"]
        height, width, _ = img.shape
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if self.keep_ratio:
            # crop (top, left, height, width)
            img = img[i:(i + h), j:(j + w), :]
            boxes -= np.array([j, i, j, i])
            boxes = clip_boxes_to_image(boxes, [h, w])

            # resize
            scale = min(self.size[0] / h, self.size[1] / w)
            ow = int(w * scale)
            oh = int(h * scale)
            img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
            # pad,  left, top, right and bottom
            padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
            padh /= 2
            padw /= 2  # divide padding into 2 sides
            top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
            left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)  # add border
            boxes[:, 1::2] = boxes[:, 1::2] * scale + top
            boxes[:, 0::2] = boxes[:, 0::2] * scale + left
            keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
            if keep.shape[0] < 1:
                img, target = sample['image'], sample['target']
                boxes = target["boxes"]
                scale = min(self.size[0] / h, self.size[1] / w)
                ow = int(w * scale)
                oh = int(h * scale)
                img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
                padh, padw = self.size[0] - oh, self.size[1] - ow  # wh padding
                padh /= 2
                padw /= 2  # divide padding into 2 sides
                top, bottom = int(round(padh - 0.1)), int(round(padh + 0.1))
                left, right = int(round(padw - 0.1)), int(round(padw + 0.1))
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.fill)
                boxes[:, 1::2] = boxes[:, 1::2] * scale + top
                boxes[:, 0::2] = boxes[:, 0::2] * scale + left
                target["boxes"] = boxes
                return {'image': img, 'target': target}

            target["labels"] = labels[keep]
            target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()
            return {'image': img, 'target': target}
        else:
            # crop
            img = img[i:(i + h), j:(j + w), :]
            boxes -= [j, i, j, i]
            # resize
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
            scale_h, scale_w = self.size[0] / h, self.size[1] / w
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            keep = remove_small_boxes(boxes, self.min_size)  # remove boxes that less than 3 pixes
            if keep.shape[0] < 1:
                img, target = sample['image'], sample['target']
                boxes = target["boxes"]
                img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
                scale_h, scale_w = self.size[0] / h, self.size[1] / w
                boxes[:, 0::2] = boxes[:, 0::2] * scale_w
                boxes[:, 1::2] = boxes[:, 1::2] * scale_h
                target["boxes"] = boxes
                return {'image': img, 'target': target}

            target["labels"] = labels[keep]
            target["boxes"] = boxes[keep].astype(np.float32)  # boxes.numpy()
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
            image = Image.fromarray(img.astype(np.uint8))
            image = T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(image)
            return {'image': np.asarray(image), 'target': target}
        return {'image': img, 'target': target}


class ColorHSV(object):
    def __init__(self, p=0.5, hue=0, saturation=0, value=0):
        self.p = p
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            r = np.random.uniform(-1, 1, 3) * [self.hue, self.saturation, self.value] + 1  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            return {'image': cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img), 'target': target}
        return {'image': img, 'target': target}



class RandomEqualize(object):
    def __init__(self, p=0.5, clahe=True):
        self.p = p
        self.clahe = clahe

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            if self.clahe:
                c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                yuv[:, :, 0] = c.apply(yuv[:, :, 0])
            else:
                yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
            img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            del yuv
        return {'image': img, 'target': target}


class ToPercentCoords(object):
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        height, width, _ = img.shape
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        target["boxes"] = boxes
        return {'image': img, 'target': target}


class ToCXCYWH(object):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        boxes_cp = copy.deepcopy(boxes)
        boxes_cp[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5  # x center
        boxes_cp[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5  # y center
        boxes_cp[:, 2] = (boxes[:, 2] - boxes[:, 0])  # width
        boxes_cp[:, 3] = (boxes[:, 3] - boxes[:, 1])  # height

        if self.normalize:
            height, width, _ = img.shape
            boxes_cp[:, 0::2] /= width
            boxes_cp[:, 1::2] /= height

        target["boxes"] = boxes_cp
        del boxes_cp
        return {'image': img, 'target': target}


class ToXYXY(object):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        boxes_cp = copy.deepcopy(boxes)

        boxes_cp[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5  # top left x
        boxes_cp[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5  # top left y
        boxes_cp[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5  # bottom right x
        boxes_cp[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5  # bottom right y

        if self.normalize:
            height, width, _ = img.shape
            boxes_cp[:, 0::2] /= width
            boxes_cp[:, 1::2] /= height

        target["boxes"] = boxes_cp
        del boxes_cp
        return {'image': img, 'target': target}


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


class RandomMotionBlur(object):
    def __init__(self, p=0.5, degrees=[2,3]):
        super(RandomMotionBlur, self).__init__()
        self.p = p
        self.degrees = degrees if isinstance(degrees, list) else (-degrees, degrees)

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            boxes = target["boxes"]

            degree = random.randint(self.degrees[0], self.degrees[1])
            angle = random.uniform(-360, 360)

            # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
            M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
            motion_blur_kernel = np.diag(np.ones(degree))
            motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

            motion_blur_kernel = motion_blur_kernel / degree
            blurred = cv2.filter2D(img, -1, motion_blur_kernel)

            # convert to uint8
            cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
            img = np.array(blurred, dtype=np.uint8)
            target["boxes"] = boxes
        return {'image': img, 'target': target}


class RandomRotation(object):
    def __init__(self, p=0.5, degrees=0):
        super(RandomRotation, self).__init__()
        self.p = p
        self.degrees = degrees if isinstance(degrees, list) else (-degrees, degrees)

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            boxes = target["boxes"]
            angle = random.uniform(self.degrees[0], self.degrees[1])
            h, w, _ = img.shape
            rx0, ry0 = w / 2.0, h / 2.0
            M = cv2.getRotationMatrix2D((rx0, ry0), angle, 1)
            img = cv2.warpAffine(img, M, (h, w))

            a = -angle / 180.0 * math.pi
            new_boxes = np.zeros_like(boxes)
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
                z = np.array([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
                tp = np.zeros_like(z)
                tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
                tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
                ymax, xmax = np.max(tp, 0)
                ymin, xmin = np.min(tp, 0)
                new_boxes[i] = np.stack([ymin, xmin, ymax, xmax])
            new_boxes = clip_boxes_to_image(new_boxes, (h, w))
            boxes[:, 0] = new_boxes[:, 1]
            boxes[:, 1] = new_boxes[:, 0]
            boxes[:, 2] = new_boxes[:, 3]
            boxes[:, 3] = new_boxes[:, 2]
            target["boxes"] = boxes
        return {'image': img, 'target': target}


class RandomAffine(object):
    '''torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))'''
    def __init__(self, p=0.5, degrees=[0., 0.], translate=0., scale=[0.5, 1.5], shear=[0., 0.], perspective=[0., 0.], border=[0, 0], fill= [128, 128, 128]):
        self.p = p
        self.degrees = degrees if isinstance(degrees, list) else (-degrees, degrees)
        self.translate = translate
        self.scale = scale
        self.shear = shear if isinstance(shear, list) else (-shear, shear)
        self.perspective = perspective if isinstance(perspective, list) else (-perspective, perspective)
        self.border = border if isinstance(border, list) else (border, border)
        self.fill = fill

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            boxes = target["boxes"]
            labels = target["labels"]

            height = img.shape[0] + self.border[0] * 2  # shape(h,w,c)
            width = img.shape[1] + self.border[1] * 2

            # Center
            C = np.eye(3)
            C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
            C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

            # Perspective
            P = np.eye(3)
            P[2, 0] = random.uniform(self.perspective[0], self.perspective[1])  # x perspective (about y)
            P[2, 1] = random.uniform(self.perspective[0], self.perspective[1])  # y perspective (about x)

            # Rotation and Scale
            R = np.eye(3)
            a = random.uniform(self.degrees[0], self.degrees[1])
            # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
            s = random.uniform(self.scale[0], self.scale[1])
            # s = 2 ** random.uniform(-scale, scale)
            R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

            # Shear
            S = np.eye(3)
            S[0, 1] = math.tan(random.uniform(self.shear[0], self.shear[1]) * math.pi / 180)  # x shear (deg)
            S[1, 0] = math.tan(random.uniform(self.shear[0], self.shear[1]) * math.pi / 180)  # y shear (deg)

            # Translation
            T = np.eye(3)
            T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
            T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

            # Combined rotation matrix
            M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
            if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
                if self.perspective:
                    img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=self.fill)
                else:  # affine
                    img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=self.fill)

            # Transform label coordinates
            n = len(boxes)
            if n:
                # warp boxes
                xy = np.ones((n * 4, 3))
                xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
                xy = xy @ M.T  # transform
                xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

                # create new boxes
                x = xy[:, [0, 2, 4, 6]]
                y = xy[:, [1, 3, 5, 7]]
                new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

                # clip
                new = clip_boxes_to_image(new, [height, width])

                # filter candidates
                i = box_candidates(box1=boxes.T * s, box2=new.T, area_thr=0.10)
                boxes = new[i]
                labels = labels[i]

            target["boxes"] = boxes.astype(np.float32)
            target["labels"] = labels
        return {'image': img ,'target': target}


class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return {'image': img, 'target': target}


class GaussianBlur(object):
    def __init__(self, p=0.1, ksize=[(3, 3), (5, 5), (7, 7)]):
        self.p = p
        self.ksize = ksize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.GaussianBlur(img, random.choice(self.ksize), 0)
        return {'image': img, 'target': target}


class MedianBlur(object):
    def __init__(self, p=0.1, ksize=[3, 5, 7]):
        self.p = p
        self.ksize = ksize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            img = cv2.medianBlur(img, int(random.choice(self.ksize)))
        return {'image': img, 'target': target}


def bbox_ioa(box1, box2, eps=1E-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


class Cutout(object):
    '''
        Improved Regularization of Convolutional Neural Networks with Cutout
        https://arxiv.org/abs/1708.04552
    '''
    def __init__(self, p=0.5, alpha=32, beta=32):
        self.p = p
        self.alpha = alpha
        self.beta = beta

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            boxes = target["boxes"]
            labels = target["labels"]
            height, width, _ = img.shape
            scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
            for s in scales:
                mask_h = random.randint(1, int(height * s))  # create random masks
                mask_w = random.randint(1, int(width * s))

                # box
                xmin = max(0, random.randint(0, width) - mask_w // 2)
                ymin = max(0, random.randint(0, height) - mask_h // 2)
                xmax = min(width, xmin + mask_w)
                ymax = min(height, ymin + mask_h)

                # apply random color mask
                img[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

                # return unobscured labels
                if len(boxes) and s > 0.03:
                    box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                    ioa = bbox_ioa(box, boxes)  # intersection over area
                    boxes = boxes[ioa < 0.60]  # remove >60% obscured labels
                    labels = labels[ioa < 0.60]

            target["boxes"] = boxes
            target["labels"] = labels
        return {'image': img, 'target': target}


class Mixup(object):
    '''
        mixup: BEYOND EMPIRICAL RISK MINIMIZATIONmixup: BEYOND EMPIRICAL RISK MINIMIZATION
        https://arxiv.org/pdf/1710.09412.pdf
    '''
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        assert len(sample) == 2, 'the number of input must eq. 2'
        img0, target0 = sample[0]['image'], sample[0]['target']
        img1, target1 = sample[1]['image'], sample[1]['target']

        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        img = (img0 * r + img1 * (1 - r)).astype(np.uint8)

        target = {}
        target["height"] = target0["height"]
        target["width"] = target0["width"]
        target["boxes"] = np.concatenate((target0["boxes"], target1["boxes"]), 0)
        target["labels"] = torch.tensor(np.concatenate((target0["labels"], target1["labels"]), 0))
        return {'image': img, 'target': target}


class CopyPaste(object):
    '''
        Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation
        https://arxiv.org/abs/2012.07177
    '''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        return {'image': img, 'target': target}


class Mosaic(object):
    def __init__(self, p, size, fill=128, min_size=3):
        self.p = p
        self.size = size # h, w
        self.fill = fill
        self.min_size = min_size

    def __call__(self, sample):
        assert len(sample) == 4, 'the number of input must eq. 4'
        labels4 = []
        boxes4 = []
        yc, xc = [int(random.uniform(x * 0.5, x * 1.5)) for x in self.size]
        for i, sp in enumerate(sample):
            img_t, target_t = sp['image'], sp['target']
            boxes_t = target_t["boxes"]
            h0, w0, c = img_t.shape

            scale = min(self.size[0] / h0, self.size[1] / w0)
            h, w = int(round(h0 * scale)), int(round(w0 * scale))
            if (h0 != h) or (w0 != w):
                img_t = cv2.resize(img_t, (w, h), interpolation=cv2.INTER_LINEAR)
            boxes_t *= scale

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((self.size[0] * 2, self.size[1] * 2, c), self.fill, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.size[0] * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.size[1] * 2), min(self.size[0] * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img_t[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes_t[:, 0::2] += padw
            boxes_t[:, 1::2] += padh

            # Labels
            boxes4.append(boxes_t)
            labels4.append(target_t["labels"])
        # Concat/clip labels
        boxes4 = np.concatenate(boxes4, 0)
        labels4 = np.concatenate(labels4, 0)

        # clip when using random_perspective()
        # boxes4 = clip_boxes_to_image(boxes4, self.size)
        # keep4 = remove_small_boxes(boxes4, self.min_size) # remove boxes that less than 3 pixes

        target = {}
        target["height"] = torch.tensor(self.size[0] * 2)
        target["width"] = torch.tensor(self.size[1] * 2)
        target["boxes"] = boxes4
        target["labels"] = torch.tensor(labels4)
        return {'image': img4, 'target': target}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    n = len(targets)
    if n:
        new = np.zeros((n, 4))
        # warp boxes
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        # clip_coords(x, (h - eps, w - eps))  # warning: inplace clip
        x[:, [0, 2]] = x[:, [0, 2]].clip(0, w - eps)  # x1, x2
        x[:, [1, 3]] = x[:, [1, 3]].clip(0, h - eps)  # y1, y2
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


class YOLOv5AugWithoutMosaic(object):
    def __init__(self, augment=True, img_size=640):
        self.augment = augment
        self.img_size = img_size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        labels = target["labels"]

        # load_image + letterbox
        h0, w0 = img.shape[:2]  # orig hw
        r = 640 / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        # return img, (h0, w0), img.shape[:2]  # im, hw_original, hw_resized
        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)

        lbls = np.hstack((labels, boxes))
        img, lbls = random_perspective(img, lbls,degrees=0,translate=0.1,scale=0.5,shear=0,perspective=0.0)
        labels, boxes = lbls[:,0], lbls[:,1:]

        target["boxes"] = boxes
        return {'image': img, 'target': target}




class YOLOv5AugWithoutAugment(object):
    def __init__(self, augment=True, img_size=640):
        self.augment = augment
        self.img_size = img_size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        labels = target["labels"]

        # load_image + letterbox
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        # return img, (h0, w0), img.shape[:2]  # im, hw_original, hw_resized
        img, ratio, pad = letterbox(img, self.img_size, auto=False, scaleup=self.augment)

        boxes[:, 0::2] = boxes[:, 0::2] * r + pad[0]
        boxes[:, 1::2] = boxes[:, 1::2] * r + pad[1]
        boxes = xyxy2xywhn(boxes, w=img.shape[1], h=img.shape[0], clip=True, eps=1E-3)

        target["pads"] = torch.tensor([pad[1], pad[0]], dtype=torch.float)
        target["scales"] = torch.tensor([r, r], dtype=torch.float)
        target["labels"] = labels
        target["boxes"] = boxes
        return {'image': img, 'target': target}


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, sample):
        if isinstance(sample, list):
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
            return sample
        else:
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
        # mask = torch.as_tensor(mask, dtype=torch.uint8)
        # mask = mask.any(dim=2)
        mask = np.any(mask, 2)
        masks.append(mask)
    if masks:
        # masks = torch.stack(masks, dim=0)
        masks = np.stack(masks, 0)
    else:
        # masks = torch.zeros((0, height, width), dtype=torch.uint8)
        masks = np.zeros((0, height, width), dtype=np.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, use_mask=False, use_keypoints=False):
        self.use_mask = use_mask
        self.use_keypoints = use_keypoints

    def __call__(self, sample):
        if isinstance(sample, list):
            for i, s in enumerate(sample):
                sample[i] = self(sample[i])
            return sample
        else:
            img, target = sample['image'], sample['target']
            h, w, _ = img.shape

            image_id = target["image_id"]
            anno = target["annotations"]

            anno = [obj for obj in anno if obj['iscrowd'] == 0]

            boxes = [obj["bbox"] for obj in anno]
            # guard against no boxes via resizing
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            boxes[:, 2:] += boxes[:, :2]  # xywh ==> xyxy
            boxes = clip_boxes_to_image(boxes, [h, w])

            labels = [obj["category_id"] for obj in anno]
            labels = torch.tensor(labels, dtype=torch.int64)

            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            labels = labels[keep]

            target = {}
            target["image_id"] = torch.tensor([image_id])
            target["height"] = torch.tensor(h)
            target["width"] = torch.tensor(w)
            target["boxes"] = boxes
            target["labels"] = labels
            if self.use_mask:
                segmentations = [obj["segmentation"] for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                target["masks"] = masks[keep]
            if self.use_keypoints:
                keypoints = None
                if anno and "keypoints" in anno[0]:
                    keypoints = [obj["keypoints"] for obj in anno]
                    keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
                    num_keypoints = keypoints.shape[0]
                    if num_keypoints:
                        keypoints = keypoints.view(num_keypoints, -1, 3)
                if keypoints is not None:
                    keypoints = keypoints[keep]
                    target["keypoints"] = keypoints

            # for conversion to coco api
            # area = torch.tensor([obj["area"] for obj in anno])
            # iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

            # target["area"] = area
            # target["iscrowd"] = iscrowd
            return {'image': img, 'target': target}


if __name__ == '__main__':
    import torch

    # trf = Resize(size=[640, 640], keep_ratio=True)
    # trf = Cutout(p=1)
    # trf = RandomGrayscale(p=1)
    # trf = RandomHorizontalFlip(p=1)
    # trf = RandomVerticalFlip(p=1)
    # trf = ColorHSV(p=1, hue=0.5, saturation=0.5, brightness=0.5)
    trf = RandomAffine(degrees=[5, 5], translate=0.1, scale=[1.5, 1.5], shear=[0., 0.], perspective=[0., 0.], border=[0, 0])
    # trf = RandomResizedCrop(size=[320, 320], scale=[0.6, 1.4], ratio=[0.5, 2.0], keep_ratio=True)
    # trf = RandomCrop(size=[320, 320])

    sample = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/sample1.pth')
    # print(sample)

    '''
    img, target = sample['image'], sample['target']
    boxes = target["boxes"]
    for box in boxes:
        cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), 1, 0)
    cv2.imwrite('img.jpg', img)
    '''

    sample1 = trf(sample)
    img1, target1 = sample1['image'], sample1['target']
    boxes1 = target1["boxes"]
    for box in boxes1:
        cv2.rectangle(img1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 1, 0)
    cv2.imwrite('img2.jpg', img1)

    import cv2
    import numpy as np
    import copy

    # torch.save(imgs, '/home/lmin/pythonCode/scripts/weights/ssd/imgs.pth')
    # torch.save(targets, '/home/lmin/pythonCode/scripts/weights/ssd/targets.pth')

    imgs = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/imgs.pth')
    targets = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/targets.pth')

    for i, (img, target) in enumerate(zip(imgs, targets)):
        img = img.cpu().numpy().transpose((1, 2, 0))
        img = np.ascontiguousarray(img)
        height = target['height'].cpu().numpy()
        width = target['width'].cpu().numpy()
        for bb in target['boxes']:
            bb = bb.cpu().numpy()
            '''
            bb_cp = copy.deepcopy(bb)
            bb_cp[0] = bb[0] - bb[2] * 0.5  # top left x
            bb_cp[1] = bb[1] - bb[3] * 0.5  # top left y
            bb_cp[2] = bb[0] + bb[2] * 0.5  # bottom right x
            bb_cp[3] = bb[1] + bb[3] * 0.5  # bottom right y
            bb_cp[0::2] *= width
            bb_cp[1::2] *= height
            img = img.astype(np.uint8)
            cv2.rectangle(img, (int(bb_cp[0]), int(bb_cp[1])), (int(bb_cp[2]), int(bb_cp[3])), (0, 0, 255), 1, 0)
            '''
            img = img.astype(np.uint8)
            cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 1, 0)
        # cv2.imwrite('aaa_' + str(i) + '.jpg', img)
        assert 1 == 2
