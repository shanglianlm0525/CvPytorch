# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/8/9 8:51
# @Author : liumin
# @File : array_ops_test.py

import numpy as np


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



if __name__=="__main__":
    boxes = np.array([ [ 2.06739990e+02,  2.20799866e+01,  2.43400024e+02,  1.37899994e+02],
 [ 2.38699951e+01, -5.19989014e-01,  5.74500122e+01,  1.02450012e+02],
 [ 8.48900146e+01,  2.21199951e+01,  1.39489990e+02,  1.45619995e+02],
 [ 1.71429993e+02, -6.20599976e+01,  2.20020020e+02, -2.89100037e+01],
 [ 3.20670044e+02, -1.34529999e+02,  3.38840027e+02, -1.12639999e+02],
 [-2.21799927e+01, -1.17000000e+02, -5.36999512e+00, -8.11999969e+01],
 [-5.02099915e+01, -9.40200043e+01, -3.82500000e+01, -8.40599976e+01],
 [-5.25100098e+01, -7.83300018e+01, -3.48399963e+01, -5.51199951e+01],
 [ 5.08999634e+00, -1.82699890e+01,  3.12600098e+01,  5.46700134e+01],
 [ 4.13859985e+02, -7.62700043e+01,  4.24119995e+02, -3.56799927e+01]])
    min_size = 20

    print(boxes)
    keep = remove_small_boxes(boxes, min_size)
    print(keep)
    print(boxes[keep])
    boxes = clip_boxes_to_image(boxes, [640, 640])
    print(boxes)
