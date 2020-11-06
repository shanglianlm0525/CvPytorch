# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/5 10:46
# @Author : liumin
# @File : fcos_detect.py

import torch
import torch.nn as nn

from ..backbones.resnet import ResNet
from ..heads.fcos_head import ClsCntRegHead
from ..necks.fpn import FPN


def coords_fmap2orig(feature, stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns
    coords [n,2]
    '''
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class FcosBody(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ResNet(backbone='resnet50', backbone_path=None, use_fpn=True)
        self.fpn = FPN(features=256)
        self.head = ClsCntRegHead(256, num_classes, 0.01)

    def train(self, mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False


        self.apply(freeze_bn)
        self.backbone.freeze_stages(1)


    def forward(self, x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]


class GenTargets(nn.Module):
    def __init__(self):
        super().__init__()
        self.strides = [8, 16, 32, 64, 128]
        self.limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
        assert len(self.strides) == len(self.limit_range)

    def forward(self, inputs):
        '''
        inputs
        [0]list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        gt_boxes [batch_size,m,4]
        classes [batch_size,m]
        stride int
        limit_range list [min,max]
        Returns
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)  # [h*w,2]

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)  # [batch_size,h*w,m,4]

        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]

        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu

        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size,h*w,m]

        areas[~mask_pos] = 99999999
        areas_min_ind = torch.min(areas, dim=-1)[1]  # [batch_size,h*w]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
                                                                                  1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))  # [batch_size,h*w,4]

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
        cls_targets = classes[
            torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)  # [batch_size,h*w,1]

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        return cls_targets, cnt_targets, reg_targets

class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,batch_imgs,batch_boxes):
        batch_boxes=batch_boxes.clamp_(min=0)
        h,w=batch_imgs.shape[2:]
        batch_boxes[...,[0,2]]=batch_boxes[...,[0,2]].clamp_(max=w-1)
        batch_boxes[...,[1,3]]=batch_boxes[...,[1,3]].clamp_(max=h-1)
        return batch_boxes

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
            idx = (iou <= thr).nonzero().squeeze()
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