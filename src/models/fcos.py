# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/3/23 11:13
# @Author : liumin
# @File : fcos.py
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.backbones import build_backbone

from src.models.fcos_loss import LOSS, GenTargets

"""
    FCOS: Fully Convolutional One-Stage Object Detection
    https://arxiv.org/pdf/1904.01355.pdf
"""

__all__ = ["FCOS"]


class FPN(nn.Module):
    '''only for resnet50,101,152'''

    def __init__(self, features=256, use_p5=True):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(2048, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(1024, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(512, features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(2048, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]


class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, class_num, GN=True, cnt_on_reg=True, prior=0.01):
        '''
        Args
        in_channel
        class_num
        GN
        prior
        '''
        super(ClsCntRegHead, self).__init__()
        self.prior = prior
        self.class_num = class_num
        self.cnt_on_reg = cnt_on_reg

        cls_branch = []
        reg_branch = []

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32, in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32, in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv = nn.Sequential(*cls_branch)
        self.reg_conv = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(in_channel, class_num, kernel_size=3, padding=1)
        self.cnt_logits = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channel, 4, kernel_size=3, padding=1)

        self.apply(self.init_conv_RandomNormal)

        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        '''inputs:[P3~P7]'''
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds


def coords_fmap2orig(feature,stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    '''
    h,w=feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords

class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        # if self.config.add_centerness:
        cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero(as_tuple=False).squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

class FCOS(nn.Module):
    def __init__(self, dictionary=None):
        super(FCOS, self).__init__()
        self.dictionary = dictionary
        self.dummy_input = torch.zeros(1, 3, 480, 360)

        self.num_classes = len(self.dictionary)
        self.category = [v for d in self.dictionary for v in d.keys()]
        self.weight = [d[v] for d in self.dictionary for v in d.keys() if v in self.category]

        backbone_cfg = {'name': 'ResNet', 'subtype': 'resnet50', 'out_stages': [2, 3, 4], 'output_stride': 32}
        self.backbone = build_backbone(backbone_cfg)

        self.neck = FPN(256, use_p5=True)
        self.head = ClsCntRegHead(256, class_num=80, GN=True, cnt_on_reg=True, prior=0.01)

        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1,64],[64,128],[128,256],[256,512],[512,999999]]
        self.target_layer = GenTargets(strides=strides, limit_range=limit_range)
        self.loss_layer = LOSS()

        self.detection_head = DetectHead(score_threshold=0.05, nms_iou_threshold=0.6,
                                     max_detection_boxes_num=1000, strides=strides)
        self.clip_boxes = ClipBoxes()
        # self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False

        self.apply(freeze_bn)
        print("INFO===>success frozen BN")
        self.backbone.freeze_stages(1)
        print("INFO===>success frozen backbone stage1")


    def trans_specific_format(self, imgs, targets):
        batch_size=len(imgs)
        h_list = [int(img.shape[1]) for img in imgs]
        w_list = [int(img.shape[2]) for img in imgs]
        num_list = [int(target['labels'].shape[0]) for target in targets]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        max_num = np.array(num_list).max()

        pad_imgs = []
        for img in imgs:
            pad_imgs.append(
                F.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.))

        pad_boxes = []
        pad_classes = []
        for target in targets:
            pad_boxes.append(
                F.pad(target['boxes'], (0, 0, 0, max_num - target['boxes'].shape[0]), value=-1))
            pad_classes.append(
                F.pad(target['labels'], (0, max_num - target['labels'].shape[0]), value=-1))

        return torch.stack(pad_imgs), torch.stack(pad_boxes), torch.stack(pad_classes)

    def forward(self, imgs, targets=None, mode='infer', **kwargs):
        threshold = 0.05
        if mode == 'infer':
            pass
        else:
            losses = {}
            imgs, boxes, classes = self.trans_specific_format(imgs, targets)

            C3, C4, C5 = self.backbone(imgs)
            all_P = self.neck([C3, C4, C5])
            cls_logits, cnt_logits, reg_preds = self.head(all_P)
            out = [cls_logits, cnt_logits, reg_preds]
            out_targets = self.target_layer([out, boxes, classes])
            loss_tuple = self.loss_layer([out, out_targets])

            losses['cls_loss'] = loss_tuple[0]
            losses['cnt_loss'] = loss_tuple[1]
            losses['reg_loss'] = loss_tuple[2]
            losses['loss'] = loss_tuple[-1]

            if mode == 'val':
                pred_scores, pred_labels, pred_boxes = self.detection_head(out)
                pred_boxes = self.clip_boxes(imgs, pred_boxes)

                outputs = []
                for target, boxes, labels, scores in zip(targets, pred_boxes, pred_labels, pred_scores):
                    # scores are sorted, so we can break
                    keep = scores > threshold
                    outputs.append({"boxes": boxes[keep]/target['scales'], "labels": labels[keep],
                                                                "scores": scores[keep]})
                return losses, outputs
            else:
                return losses