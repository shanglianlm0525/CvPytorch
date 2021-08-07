# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/17 9:50
# @Author : liumin
# @File : det_transforms_pil.py

import torch
import copy
import random
import warnings
from collections import Sequence
from torch import Tensor
import cv2
import numbers

import math
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F, InterpolationMode
from pycocotools import mask as coco_mask
import numpy as np

from src.data.transforms.seg_transforms import _setup_size

__all__ = ['RandomHorizontalFlip', 'RandomVerticalFlip',
        'Resize',
        'RandomResizedCrop', 'RandomCrop',
        'RandomRotate', 'RandomAffine',
        'ColorJitter', 'GaussianBlur',
        'ToXYXY', 'ToXYWH', 'ToPercentCoords',
        'Normalize', 'ToTensor',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']


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

    def __init__(self, size, keep_ratio=True, fill=0, padding_mode='constant', interpolation=InterpolationMode.BILINEAR):
        # assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        if isinstance(size, numbers.Number):
            self.size = (size, size)
        else:
            self.size = size

        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self.fill = fill
        self.padding_mode = padding_mode

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

        if self.keep_ratio:
            scale = min(self.size[0] / h, self.size[1] / w)
            ow = int(w * scale)
            oh = int(h * scale)
            img = F.resize(img, [oh, ow], self.interpolation)
            # left, top, right and bottom
            img = F.pad(img, [0, 0, self.size[1] - ow, self.size[0] - oh], self.fill, self.padding_mode)
            boxes[:, :4] = boxes[:, :4] * scale

            target["boxes"] = boxes
            target["scales"] = torch.tensor([scale, scale], dtype=torch.float)
            return {'image': img, 'target': target}
        else:
            scale_h, scale_w = self.size[0] / h, self.size[1] / w
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w

            target["boxes"] = boxes
            target["scales"] = torch.tensor([scale_h, scale_w], dtype=torch.float)
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
                xmin = w-1 - boxes[:, 2]
                xmax = w-1 - boxes[:, 0]
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
                ymin = h-1 - boxes[:, 3]
                ymax = h-1 - boxes[:, 1]
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


class ToArrayImage(object):
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        return {'image': np.asarray(img, dtype=np.float32), 'target': target}


class ToPercentCoords(object):
    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        width, height = img.size
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        target["boxes"] = boxes
        return {'image': img, 'target': target}


class ToXYWH(object):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        boxes_cp = boxes.clone()
        if self.normalize:
            w, h = img.size
            boxes_cp[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / w # x center
            boxes_cp[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / h # y center
            boxes_cp[:, 2] = (boxes[:, 2] - boxes[:, 0]) / w # width
            boxes_cp[:, 3] = (boxes[:, 3] - boxes[:, 1]) / h  # height
        else:
            boxes_cp[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5  # x center
            boxes_cp[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5  # y center
            boxes_cp[:, 2] = (boxes[:, 2] - boxes[:, 0])  # width
            boxes_cp[:, 3] = (boxes[:, 3] - boxes[:, 1])  # height

        target["boxes"] = boxes_cp
        return {'image': img, 'target': target}


class ToXYXY(object):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    def __init__(self, normalize=False):
        self.normalize = normalize

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        boxes_cp = boxes.clone()
        if self.normalize:
            w, h = img.size
            boxes_cp[:, 0] = (boxes[:, 0] - boxes[:, 2] * 0.5) / w  # top left x
            boxes_cp[:, 1] = (boxes[:, 1] - boxes[:, 3] * 0.5) / h  # top left y
            boxes_cp[:, 2] = (boxes[:, 0] + boxes[:, 2] * 0.5) / w  # bottom right x
            boxes_cp[:, 3] = (boxes[:, 1] + boxes[:, 3] * 0.5) / h  # bottom right y
        else:
            boxes_cp[:, 0] = boxes[:, 0] - boxes[:, 2] * 0.5  # top left x
            boxes_cp[:, 1] = boxes[:, 1] - boxes[:, 3] * 0.5  # top left y
            boxes_cp[:, 2] = boxes[:, 0] + boxes[:, 2] * 0.5  # bottom right x
            boxes_cp[:, 3] = boxes[:, 1] + boxes[:, 3] * 0.5  # bottom right y
        target["boxes"] = boxes_cp
        return {'image': img, 'target': target}


class RandomAffine(object):
    """Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        fillcolor (sequence or number, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``fill`` parameter instead.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self, p=0.5, degrees=(0, 0), translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0):
        super().__init__()
        self.p = p
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = interpolation
        self.fill = fill

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


    def __call__(self, sample):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]

        if random.random() < self.p:
            fill = self.fill
            if isinstance(img, Tensor):
                if isinstance(fill, (int, float)):
                    fill = [float(fill)] * F._get_image_num_channels(img)
                else:
                    fill = [float(f) for f in fill]

            img_size = F._get_image_size(img)
            width, height = img_size

            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
            # print(ret)
            angle, (translation_x, translation_y), scale, (shear_x, shear_y) = ret
            # angle
            rx0, ry0 = width / 2.0, height / 2.0
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
            new_boxes[:, 1::2].clamp_(min=0, max=width - 1)
            new_boxes[:, 0::2].clamp_(min=0, max=height - 1)
            boxes[:, 0] = new_boxes[:, 1]
            boxes[:, 1] = new_boxes[:, 0]
            boxes[:, 2] = new_boxes[:, 3]
            boxes[:, 3] = new_boxes[:, 2]

            # translations
            boxes += torch.Tensor([translation_x, translation_y, translation_x, translation_y])
            # scale
            boxes *= scale
            boxes += torch.Tensor([int(width * 0.5 * (1-scale)), int(height* 0.5 * (1-scale)),
                                   int(width * 0.5 * (1-scale)), int(height* 0.5 * (1-scale))])
            # shear

            # clamp
            boxes[:, 1::2].clamp_(min=max(translation_y, 0), max=min(height+translation_y-1, height-1))
            boxes[:, 0::2].clamp_(min=max(translation_x, 0), max=min(width+translation_x-1, width-1))
            # return F.affine(img, *ret, interpolation=self.interpolation, fill=fill)
            target["boxes"] = boxes # 0.3, 0.7
            return {'image': F.affine(img, *ret, interpolation=self.interpolation, fill=fill),
                    'target': target}

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

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()

        self.size = tuple(_setup_size(
            size, error_msg="Please provide only two dimensions (h, w) for size."
        ))

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        img, target = sample['image'], sample['target']
        boxes = target["boxes"]
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        boxes -= torch.Tensor([j, i, j, i])

        boxes[:, 1::2].clamp_(min=0, max=h - 1)
        boxes[:, 0::2].clamp_(min=0, max=w - 1)

        target["boxes"] = boxes
        return {'image': F.crop(img, i, j, h, w), 'target': target}


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

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), keep_ratio=True, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.keep_ratio = keep_ratio

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
        width, height = F._get_image_size(img)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)

        if self.keep_ratio:
            # crop
            img = F.crop(img, i, j, h, w)
            boxes -= torch.Tensor([j, i, j, i])
            # resize
            scale = min(self.size[0] / h, self.size[1] / w)
            ow = int(w * scale)
            oh = int(h * scale)
            img = F.resize(img, [oh, ow], self.interpolation)
            boxes[:, :4] = boxes[:, :4] * scale
            # pad,  left, top, right and bottom
            img = F.pad(img, [0, 0, self.size[1] - ow, self.size[0] - oh], 0, 'constant')

            # clamp
            boxes[:, 1::2].clamp_(min=0, max=oh - 1)
            boxes[:, 0::2].clamp_(min=0, max=ow - 1)

            target["boxes"] = boxes  # boxes.numpy()
            return {'image': img, 'target': target}
        else:
            # crop
            boxes -= torch.Tensor([j, i, j, i])
            # resize
            scale_h, scale_w = self.size[0] / h, self.size[1] / w
            boxes[:, 0::2] = boxes[:, 0::2] * scale_w
            boxes[:, 1::2] = boxes[:, 1::2] * scale_h
            # clamp
            boxes[:, 1::2].clamp_(min=0, max=self.size[0] - 1)
            boxes[:, 0::2].clamp_(min=0, max=self.size[1] - 1)

            target["boxes"] = boxes  # boxes.numpy()
            return {'image': F.resized_crop(img, i, j, h, w, self.size, self.interpolation), 'target': target}

'''
keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
if torch.nonzero(keep).size(0) < 1:
    return 
'''



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


if __name__ == '__main__':
    import torch
    from PIL import ImageDraw

    # trf = Resize(size=[300, 300], keep_ratio=False)
    # trf = RandomScaleCrop(size=[300, 300], scale=(0.5, 1.0), keep_ratio=False)
    # trf = RandomResizedCrop(size=[300, 300], scale=(0.6, 1.0), ratio=(0.5, 2.0), keep_ratio=True)
    # trf = RandomAffine(p=1, degrees=[0.5, 2.0], keep_ratio=False)
    trf = RandomCrop(size=[300, 300])

    data = torch.load('/home/lmin/pythonCode/scripts/weights/ssd/sample.pth')
    target = {}
    anno = data['target']["annotations"]
    boxes = [obj["bbox"] for obj in anno]
    # guard against no boxes via resizing
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    boxes[:, 2:] += boxes[:, :2]
    target["boxes"] = boxes

    classes = [obj["category_id"] for obj in anno]
    classes = torch.tensor(classes, dtype=torch.int64)
    target["labels"] = classes
    sample = {'image': data['image'], 'target': target}
    im = copy.deepcopy(data['image'])
    draw = ImageDraw.Draw(im)
    # p = sample['target']['boxes'][0, :]
    for p in sample['target']['boxes']:
        draw.line([(p[0], p[1]), (p[2], p[1]), (p[2], p[3]), (p[0], p[3]), (p[0], p[1])], width=1, fill='blue')
    im.save('img.png')
    print(sample['image'].size)
    print(sample['target']['boxes'][:6,:])

    sample = trf(sample)
    print(sample['image'].size)
    sample['image'].save('img1.png')
    im1 = copy.deepcopy(sample['image'])
    draw1 = ImageDraw.Draw(im1)
    # p1 = sample['target']['boxes'][0, :]
    for p1 in sample['target']['boxes']:
        draw1.line([(p1[0], p1[1]), (p1[2], p1[1]), (p1[2], p1[3]), (p1[0], p1[3]), (p1[0], p1[1])], width=1, fill='red')
    im1.save('img2.png')
    print(sample['target']['boxes'][:5,:])
    # print(sample['target']['scales'])
