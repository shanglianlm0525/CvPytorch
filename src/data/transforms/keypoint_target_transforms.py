# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/9/18 15:39
# @Author : liumin
# @File : keypoint_target_transforms.py

import numpy as np


__all__ = ['Compose', 'OpenPoseTargetTransform']

import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',
        'left_eye',
        'right_ear',
        'left_ear']

    return keypoints

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_hip')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('neck'), keypoints.index('left_hip')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],
        [keypoints.index('neck'), keypoints.index('left_shoulder')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],
        [keypoints.index('neck'), keypoints.index('nose')],
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]
    return kp_lines

def remove_illegal_joint(keypoints, input_x, input_y):
    MAGIC_CONSTANT = (-1, -1, 0)
    mask = np.logical_or.reduce((keypoints[:, :, 0] >= input_x,
                                 keypoints[:, :, 0] < 0,
                                 keypoints[:, :, 1] >= input_y,
                                 keypoints[:, :, 1] < 0))
    keypoints[mask] = MAGIC_CONSTANT
    return keypoints


def add_neck(keypoint):
    '''
    MS COCO annotation order:
    0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
    5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
    9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
    14: r knee		15: l ankle		16: r ankle
    The order in this work:
    (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
    5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
    9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
    13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
    17-'left_ear' )
    '''
    our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
    # Index 6 is right shoulder and Index 5 is left shoulder
    right_shoulder = keypoint[6, :]
    left_shoulder = keypoint[5, :]
    neck = (right_shoulder + left_shoulder) / 2
    if right_shoulder[2] == 2 and left_shoulder[2] == 2:
        neck[2] = 2
    else:
        neck[2] = right_shoulder[2] * left_shoulder[2]

    neck = neck.reshape(1, len(neck))
    neck = np.round(neck)
    keypoint = np.vstack((keypoint, neck))
    keypoint = keypoint[our_order, :]
    return keypoint


def putGaussianMaps(center, accumulate_confid_map, sigma, grid_y, grid_x, stride):
    """Implement the generate of every channel of ground truth heatmap.
    :param centerA: int with shape (2,), every coordinate of person's keypoint.
    :param accumulate_confid_map: one channel of heatmap, which is accumulated,
           np.log(100) is the max value of heatmap.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x
    """
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map


def putVecMaps(centerA, centerB, accumulate_vec_map, count, grid_y, grid_x, stride):
    """Implement Part Affinity Fields
    :param centerA: int with shape (2,), centerA will pointed by centerB.
    :param centerB: int with shape (2,), centerB will point to centerA.
    :param accumulate_vec_map: one channel of paf.
    :param count: store how many pafs overlaped in one coordinate of accumulate_vec_map.
    :param params_transform: store the value of stride and crop_szie_y, crop_size_x
    """
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    thre = 1  # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0

    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0
    return accumulate_vec_map, count


def get_openpose_ground_truth(anns, input_x, input_y, stride):
    keypoints = []
    for single_keypoints in anns:
        # single_keypoints = np.array(ann['keypoints']).reshape(17, 3)
        single_keypoints = add_neck(single_keypoints)
        keypoints.append(single_keypoints)
    keypoints = np.array(keypoints)
    keypoints = remove_illegal_joint(keypoints, input_x, input_y)

    grid_y = int(input_y / stride)
    grid_x = int(input_x / stride)

    KEYPOINTS = get_keypoints()
    HEATMAP_COUNT = len(KEYPOINTS)
    LIMB_IDS = kp_connections(KEYPOINTS)

    channels_heat = (HEATMAP_COUNT + 1)
    channels_paf = 2 * len(LIMB_IDS)
    heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
    pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

    # confidance maps for body parts
    for i in range(HEATMAP_COUNT):
        joints = [jo[i] for jo in keypoints]
        for joint in joints:
            if joint[2] > 0.5:
                center = joint[:2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(center, gaussian_map, 7.0, grid_y, grid_x, stride)

    # pafs
    for i, (k1, k2) in enumerate(LIMB_IDS):
        # limb
        count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
        for joint in keypoints:
            if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                centerA = joint[k1, :2]
                centerB = joint[k2, :2]
                vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                    centerA=centerA, centerB=centerB, accumulate_vec_map=vec_map,
                    count=count, grid_y=grid_y, grid_x=grid_x, stride=stride)

    # background
    heatmaps[:, :, -1] = np.maximum(1 - np.max(heatmaps[:, :, :HEATMAP_COUNT], axis=2), 0.)
    return heatmaps, pafs  # [46, 46, 38], [46, 46, 38]


class OpenPoseTargetTransform(object):
    def __init__(self, input_x=368, input_y=368, stride=8):
        super(OpenPoseTargetTransform, self).__init__()
        self.input_x = input_x
        self.input_y = input_y
        self.stride = stride

    def __call__(self, sample):
        img, target = sample['image'], sample['target']

        heatmaps, pafs = get_openpose_ground_truth(target["keypoints"], self.input_x, self.input_y, self.stride)
        # torch.from_numpy(target["keypoints"])

        target["heatmaps"] = torch.from_numpy(heatmaps.transpose((2, 0, 1)).astype(np.float32))
        target["pafs"] = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))
        target.pop("keypoints")
        return {'image': img, 'target': target}