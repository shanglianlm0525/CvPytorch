# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/18 18:34
# @Author : liumin
# @File : seg_transforms.py
import copy
import warnings

import math
import torch
from torch import Tensor
import torchvision
from typing import Tuple, List, Optional
from torchvision import transforms as T
import torchvision.transforms.functional as F
from pycocotools import mask as coco_mask
import random
import numbers
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import collections
from collections.abc import Sequence
from torchvision.transforms.functional import InterpolationMode, _interpolation_modes_from_int


__all__ = ['Compose', 'ToTensor', 'Normalize',
        'RandomHorizontalFlip', 'RandomVerticalFlip',
        'RandomScale',
        'Resize', 'RandomResizedCrop',
        'RandomCrop', 'CenterCrop',
        'RandomRotation', 'RandomPerspective',
        'ColorJitter', 'GaussianBlur',
        'Pad', 'Lambda', 'Encrypt',
        'FilterAndRemapCocoCategories', 'ConvertCocoPolysToMask']


from torchvision.transforms.transforms import _setup_angle, _check_sequence_input

_pil_interpolation_to_str = {
    InterpolationMode.NEAREST: 'InterpolationMode.NEAREST',
    InterpolationMode.BILINEAR: 'InterpolationMode.BILINEAR',
    InterpolationMode.BICUBIC: 'InterpolationMode.BICUBIC',
    InterpolationMode.LANCZOS: 'InterpolationMode.LANCZOS',
}


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            img, target = sample['image'], sample['target']
            # print(t, np.asarray(target, dtype=np.uint8).shape, np.asarray(target, dtype=np.uint8))
            sample = t(sample)
            # print(t, np.asarray(target, dtype=np.uint8).shape, np.asarray(target, dtype=np.uint8))
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type

    def __call__(self, sample):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            img (PIL Image or numpy.ndarray): Image to be converted to tensor.
            target (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        img, target = sample['image'], sample['target']
        if self.normalize:
            return {'image': F.to_tensor(img), 'target': torch.from_numpy( np.array( target, dtype=self.target_type) )}
        else:
            return {'image': torch.from_numpy( np.array(img, dtype=np.float32).transpose(2, 0, 1) ),
                    'target': torch.from_numpy( np.array( target, dtype=self.target_type) )}

    def __repr__(self):
        return self.__class__.__name__ + '()'



class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        img, target = sample['image'], sample['target']
        return {'image': F.normalize(img, self.mean, self.std),'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': F.hflip(img), 'target': F.hflip(target)}
        return {'image': img, 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be flipped.
            target (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': F.vflip(img), 'target': F.vflip(target)}
        return {'image': img, 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomScale(object):
    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        img, target = sample['image'], sample['target']
        assert img.size == target.size
        scale = random.uniform(self.scale[0], self.scale[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return {'image': F.resize(img, target_size, self.interpolation), 'target': F.resize(target, target_size, InterpolationMode.NEAREST)}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(scale_range={0}, interpolation={1})'.format(self.scale, interpolate_str)


class CenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        img, target = sample['image'], sample['target']
        return {'image': F.center_crop(img, self.size), 'target': F.center_crop(target, self.size)}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=True, fill=0, ignore_label=255, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.ignore_label = ignore_label
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        # assert img.size == target.size, 'size of img and lbl should be the same. %s, %s'%(img.size, target.size)
        if self.padding > 0:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(target, self.padding, self.ignore_label, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, int((1 + self.size[1] - img.size[0]) / 2), self.fill, self.padding_mode)
            target = F.pad(target, int((1 + self.size[1] - target.size[0]) / 2), self.ignore_label, self.padding_mode)

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, int((1 + self.size[0] - img.size[1]) / 2), self.fill, self.padding_mode)
            target = F.pad(target, int((1 + self.size[0] - target.size[1]) / 2), self.ignore_label, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return {'image': F.crop(img, i, j, h, w), 'target': F.crop(target, i, j, h, w)}

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


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

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        img, target = sample['image'], sample['target']
        return {'image': F.resize(img, self.size, self.interpolation), 'target': F.resize(target, self.size, InterpolationMode.NEAREST)}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size (int or sequence): expected output size of each edge. If size is an
            int instead of sequence like (h, w), a square output size ``(size, size)`` is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        scale (tuple of float): range of size of the origin size cropped
        ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped.
        interpolation (int): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
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
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return {'image': F.resized_crop(img, i, j, h, w, self.size, self.interpolation),
                'target': F.resized_crop(target, i, j, h, w, self.size, InterpolationMode.NEAREST)}

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomRotation(torch.nn.Module):
    """Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(
        self, degrees, interpolation=F.InterpolationMode.NEAREST, expand=False, center=None, fill=0, resample=None
    ):
        super().__init__()
        if resample is not None:
            warnings.warn(
                "Argument resample is deprecated and will be removed since v0.10.0. Please, use interpolation instead"
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2, ))

        self.center = center

        self.resample = self.interpolation = interpolation
        self.expand = expand

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fill = fill

    @staticmethod
    def get_params(degrees: List[float]) -> float:
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            float: angle parameter to be passed to ``rotate`` for random rotation.
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Rotated image.
        """
        img, target = sample['image'], sample['target']
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F._get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]
        angle = self.get_params(self.degrees)

        return {'image': F.rotate(img, angle, self.resample, self.expand, self.center, fill),
                'target': F.rotate(target, angle, self.resample, self.expand, self.center)}


    def __repr__(self):
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', interpolation={0}'.format(interpolate_str)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        if self.fill is not None:
            format_string += ', fill={0}'.format(self.fill)
        format_string += ')'
        return format_string


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, p=0.5, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        img, target = sample['image'], sample['target']
        if random.random() < self.p:
            return {'image': T.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)(img), 'target': target}
        return {'image': img, 'target': target}


    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class Pad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w = img.size
        ph = (h // self.diviser + 1) * self.diviser - h if h % self.diviser != 0 else 0
        pw = (w // self.diviser + 1) * self.diviser - w if w % self.diviser != 0 else 0
        img = F.pad(img, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        target = F.pad(target, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        return {'image': img, 'target': target}


class Grayscale(object):
    """Convert image to grayscale.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
         - If ``num_output_channels == 1`` : returned image is single channel
         - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Grayscaled image.
        """
        img, target = sample['image'], sample['target']
        return {'image': F.rgb_to_grayscale(img, num_output_channels=self.num_output_channels), 'target': target}

    def __repr__(self):
        return self.__class__.__name__ + '(num_output_channels={0})'.format(self.num_output_channels)


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

    def __init__(self, p=0.5, kernel_size=[3, 3], sigma=(0.1, 2.0)):
        super().__init__()
        self.p = p
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


    def __repr__(self):
        s = '(kernel_size={}, '.format(self.kernel_size)
        s += 'sigma={})'.format(self.sigma)
        return self.__class__.__name__ + s


class RandomPerspective(object):
    """Performs a random perspective transformation of the given image with a given probability.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (int): Interpolation type. If input is Tensor, only ``PIL.Image.NEAREST`` and
            ``PIL.Image.BILINEAR`` are supported. Default, ``PIL.Image.BILINEAR`` for PIL images and Tensors.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively. Default is 0.
            This option is only available for ``pillow>=5.0.0``. This option is not supported for Tensor
            input. Fill value for the area outside the transform in the output image is always 0.

    """

    def __init__(self, distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0, ignore_label=255):
        super().__init__()
        self.p = p
        self.interpolation = interpolation
        self.distortion_scale = distortion_scale
        self.fill = fill
        self.ignore_label = ignore_label

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be Perspectively transformed.

        Returns:
            PIL Image or Tensor: Randomly transformed image.
        """
        img, target = sample['image'], sample['target']
        if torch.rand(1) < self.p:
            width, height = F._get_image_size(img)
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            return {'image': F.perspective(img, startpoints, endpoints, self.interpolation, self.fill),
             'target': F.perspective(img, startpoints, endpoints, InterpolationMode.NEAREST, self.ignore_label)}
        return {'image': img,'target': target}

    @staticmethod
    def get_params(width: int, height: int, distortion_scale: float):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width (int): width of the image.
            height (int): height of the image.
            distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = height // 2
        half_width = width // 2
        topleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        topright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(0, int(distortion_scale * half_height) + 1, size=(1, )).item())
        ]
        botright = [
            int(torch.randint(width - int(distortion_scale * half_width) - 1, width, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        botleft = [
            int(torch.randint(0, int(distortion_scale * half_width) + 1, size=(1, )).item()),
            int(torch.randint(height - int(distortion_scale * half_height) - 1, height, size=(1, )).item())
        ]
        startpoints = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class SYNC_TRANSFORM(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        self.base_size = 1024

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        w, h = img.size

        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < min(self.size):
            padh = self.size[0] - oh if oh < self.size[0] else 0
            padw = self.size[1] - ow if ow < self.size[1] else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            target = ImageOps.expand(target, border=(0, 0, padw, padh), fill=-1)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.size[1])
        y1 = random.randint(0, h - self.size[0])
        img = img.crop((x1, y1, x1 + self.size[1], y1 + self.size[0]))
        target = target.crop((x1, y1, x1 + self.size[1], y1 + self.size[0]))
        # gaussian blur as in PSP
        '''
        if random.random() < 0:
            radius = cfg.AUG.BLUR_RADIUS if cfg.AUG.BLUR_RADIUS > 0 else random.random()
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        # color jitter
        if self.color_jitter:
            img = self.color_jitter(img)
        '''
        return {'image': img,'target': target}


class VAL_TRANSFORM(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = size
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

        self.base_size = 1024

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        w, h = img.size

        outsize = self.size
        short_size = min(outsize)
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        target = target.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize[1]) / 2.))
        y1 = int(round((h - outsize[0]) / 2.))
        img = img.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))
        target = target.crop((x1, y1, x1 + outsize[1], y1 + outsize[0]))

        return {'image': img,'target': target}


class Encrypt(object):
    def __init__(self, down_size): # size: (h, w)
        self.down_size = (down_size, down_size) if isinstance(down_size, int) else down_size

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        h, w, _ = img.shape
        target = target.resize((w // self.down_size[1], h // self.down_size[0]))
        return {'image': img,'target': target}


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


####------------------- COCO Dataset ---------------#####

class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, sample):
        img, target = sample['image'], sample['target']
        anno = target["annotations"]
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
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
        anno = target["annotations"]
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return {'image': img, 'target': target}

####------------------- COCO Dataset ---------------#####