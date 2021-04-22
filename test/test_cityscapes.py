# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/18 16:43
# @Author : liumin
# @File : test_cityscapes.py
import torch
from torch.utils import data
from cityscapes import Cityscapes
import ext_transforms as et

from utils import IntermediateLayerGetter
from _deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from backbone import resnet
from backbone import mobilenetv2

'''
python main.py --model deeplabv3plus_mobilenet --dataset cityscapes 
--enable_vis --vis_port 28333 --gpu_id 0  --lr 0.1  --crop_size 768 
--batch_size 16 --output_stride 16 --data_root ./datasets/data/cityscapes 
'''


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):

    if backbone=='mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def test_city():
    crop_size = 768
    data_root = '/home/lmin/data/cityscapes/cityscapes'

    train_transform = et.ExtCompose([
        #et.ExtResize( 512 ),
        et.ExtRandomCrop(size=(crop_size, crop_size)),
        et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
        et.ExtRandomHorizontalFlip(),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    val_transform = et.ExtCompose([
        #et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])

    train_dst = Cityscapes(root=data_root,
                           split='train', transform=train_transform)
    val_dst = Cityscapes(root=data_root,
                         split='val', transform=val_transform)

    train_loader = data.DataLoader(
        train_dst, batch_size=8, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=1, shuffle=True, num_workers=2)

    model = deeplabv3plus_resnet50(num_classes=19, output_stride=16)

    for (images, labels) in train_loader:
        print(1)

# test_city()

from stream_metrics import StreamSegMetrics
import numpy as np

class SegmentationEvaluator(object):
    def __init__(self, num_classes):
        self.num_class = num_classes
        self.confusion_matrix = np.ones((self.num_class,)*2)
        self.count = 0

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Mean_Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        mAcc = np.nanmean(Acc)
        return mAcc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        print('MIoU', MIoU)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def evaluate(self):
        if self.count < 1:
            return None
        performances = {}
        performances['Acc'] = self.Pixel_Accuracy()
        performances['mAcc'] = self.Mean_Pixel_Accuracy()
        performances['mIoU'] = self.Mean_Intersection_over_Union()
        performances['FWIoU'] = self.Frequency_Weighted_Intersection_over_Union()
        performances['performance'] = performances['mIoU']
        return performances

    def update(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        gt_image = gt_image.data.cpu().numpy()
        pre_image = pre_image.data.cpu().numpy()
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)
        self.count = self.count + 1

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def test_val_city():
    data_root = '/home/lmin/data/cityscapes/cityscapes'

    val_transform = et.ExtCompose([
        # et.ExtResize( 512 ),
        et.ExtToTensor(),
        et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    val_dst = Cityscapes(root=data_root,
                         split='val', transform=val_transform)

    val_loader = data.DataLoader(
        val_dst, batch_size=1, shuffle=True, num_workers=2)

    model = deeplabv3plus_mobilenet(num_classes=19, output_stride=16)
    checkpoint = torch.load('best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model_state"])
    model.to(torch.device('cuda:0'))

    metrics = StreamSegMetrics(19)
    Evaluator = SegmentationEvaluator(19)
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(torch.device('cuda:0'), dtype=torch.float32)
            labels = labels.to(torch.device('cuda:0'), dtype=torch.long)

            outputs = model(images)
            outputs1 = outputs.detach().max(dim=1)[1]
            preds = outputs1.cpu().numpy()
            targets = labels.cpu().numpy()
            print(i, preds.shape, targets.shape)
            Evaluator.update(labels, outputs1)
            metrics.update(targets, preds)


    pr = Evaluator.evaluate()
    score = metrics.get_results()
    print(pr)
    print(score)

test_val_city()