# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/11/5 10:46
# @Author : liumin
# @File : fcos_detect.py

import torch
import torch.nn as nn

from ..backbones import build_backbone
from ..backbones.resnet import ResNet
from ..heads import build_head
from ..heads.fcos_head import FcosHead
from ..necks import build_neck
from ..necks.fcos_fpn import FPN


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
    return coords.cuda() # to(device=feature.device)


class FcosBody(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone_cfg = {'name': 'ResNet', 'subtype': 'resnet50', 'out_stages': [2, 3, 4]}
        neck_cfg = {'name': 'FPN', 'in_channels': [512, 1024, 2048], 'out_channels': 256, 'add_extra_levels':True, 'extra_levels':2}
        head_cfg = {'name': 'FcosHead', 'num_classes': 80, 'in_channel': 256, 'prior': 0.01,'cnt_on_reg': True}

        self.backbone = build_backbone(backbone_cfg)# ResNet(backbone='resnet50', backbone_path=None, use_fpn=True)
        self.fpn = FPN(features=256) # build_neck(neck_cfg) #
        self.head = build_head(head_cfg) # FcosHead(256, num_classes, 0.01)

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
        [2]gt_labels [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        gt_labels = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, gt_labels, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        cls_targets_all_level_new = [torch.empty_like(x) for x in cls_targets_all_level[0]]
        cnt_targets_all_level_new = [torch.empty_like(x) for x in cnt_targets_all_level[0]]
        reg_targets_all_level_new = [torch.empty_like(x) for x in reg_targets_all_level[0]]
        for i,(cls_targets,cnt_targets,reg_targets)  in enumerate(zip(cls_targets_all_level,cnt_targets_all_level,reg_targets_all_level)):
            for j, (cls_batch,cnt_batch,reg_batch) in enumerate(zip(cls_targets,cnt_targets,reg_targets)):
                if i == 0:
                    cls_targets_all_level_new[j] = cls_batch
                    cnt_targets_all_level_new[j] = cnt_batch
                    reg_targets_all_level_new[j] = reg_batch
                else:
                    cls_targets_all_level_new[j] = torch.cat([cls_targets_all_level_new[j], cls_batch], dim=0)
                    cnt_targets_all_level_new[j] = torch.cat([cnt_targets_all_level_new[j], cnt_batch], dim=0)
                    reg_targets_all_level_new[j] = torch.cat([reg_targets_all_level_new[j], reg_batch], dim=0)

        return cls_targets_all_level_new, cnt_targets_all_level_new, reg_targets_all_level_new

    def _gen_level_targets(self, out, gt_boxes, gt_labels, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        gt_boxes [batch_size,m,4]
        gt_labels [batch_size,m]
        stride int
        limit_range list [min,max]
        Returns
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes[0].device)  # [h*w,2]

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]

        l_off_list = [(x[None, :, None] - gt_box[..., 0]).squeeze(0) for gt_box in gt_boxes] # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off_list = [(y[None, :, None] - gt_box[..., 1]).squeeze(0) for gt_box in gt_boxes]
        r_off_list = [(gt_box[..., 2]- x[None, :, None]).squeeze(0) for gt_box in gt_boxes]
        b_off_list = [(gt_box[..., 3] - y[None, :, None]).squeeze(0) for gt_box in gt_boxes]

        ltrb_off_list = [torch.stack([l, t, r, b], dim=-1) for l, t, r, b in zip(l_off_list, t_off_list, r_off_list, b_off_list)] # [batch_size,h*w,m,4]
        areas = [(l+r)*(t+b) for l,t,r,b in zip(l_off_list,t_off_list,r_off_list,b_off_list)]

        off_min = [torch.min(t, dim=-1)[0] if t.shape[1] > 0 else torch.zeros(t.shape[:2]) for t in ltrb_off_list] # [batch_size,h*w,m]
        off_max = [torch.max(t, dim=-1)[0] if t.shape[1] > 0 else torch.zeros(t.shape[:2]) for t in ltrb_off_list] # [batch_size,h*w,m]

        mask_in_gtboxes = [off > 0 for off in off_min]
        mask_in_level = [(off > limit_range[0]) & (off <= limit_range[1]) for off in off_max]

        radiu = stride * sample_radiu_ratio
        gt_center_x = [(gt_box[..., 0] + gt_box[..., 2]) * 0.5 for gt_box in gt_boxes]
        gt_center_y = [(gt_box[..., 1] + gt_box[..., 3]) * 0.5 for gt_box in gt_boxes]

        c_l_off_list = [(x[None, :, None] - center_x[None, None, :]).squeeze(0) for center_x in gt_center_x]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off_list = [(y[None, :, None] - center_y[None, None, :]).squeeze(0) for center_y in gt_center_y]
        c_r_off_list = [(center_x[None, None, :] - x[None, :, None]).squeeze(0) for center_x in gt_center_x]
        c_b_off_list = [(center_y[None, None, :] - y[None, :, None]).squeeze(0) for center_y in gt_center_y]

        c_ltrb_off_list = [torch.stack([c_l, c_t, c_r, c_b], dim=-1) for c_l, c_t, c_r, c_b in
                    zip(c_l_off_list, c_t_off_list, c_r_off_list, c_b_off_list)]  # [batch_size,h*w,m,4]
        c_off_max = [torch.max(c_t, dim=-1)[0] if c_t.shape[1] > 0 else torch.zeros(c_t.shape[:2]) for c_t in c_ltrb_off_list]  # [batch_size,h*w,m]
        mask_center = [c_off < radiu for c_off in c_off_max]

        mask_pos = [m_gtboxes & m_level & m_center for m_gtboxes, m_level, m_center in
                    zip(mask_in_gtboxes, mask_in_level, mask_center)]   # [batch_size,h*w,m]

        for area, m_pos in zip(areas, mask_pos):
            area[~m_pos] = 99999999
        areas_min_ind = [torch.min(area, dim=-1)[1] if area.shape[1] > 0 else torch.zeros(area.shape[:2]) for area in
                         areas]

        reg_targets = [ltrb_off[torch.zeros_like(area, dtype=torch.bool).scatter_(-1, area_min_ind.unsqueeze(dim=-1), 1)]
                       for ltrb_off, area, area_min_ind in zip(ltrb_off_list, areas, areas_min_ind)]  # [batch_size*h*w,4]
        reg_targets = [torch.reshape(reg_target, (-1, 4)) for reg_target in reg_targets]  # [batch_size,h*w,4]

        classes = [torch.broadcast_tensors(cls, area.long())[0] for cls, area in zip(gt_labels, areas)] # [batch_size,h*w,m]
        cls_targets = [cls[torch.zeros_like(area, dtype=torch.bool).scatter_(-1, area_min_ind.unsqueeze(dim=-1), 1)]
                       for cls, area, area_min_ind in zip(classes, areas, areas_min_ind)]
        cls_targets = [torch.reshape(cls_target, (-1, 1)) for cls_target in cls_targets]  # [batch_size,h*w,4]

        left_right_min_list = [torch.min(reg_target[..., 0], reg_target[..., 2]) for reg_target in
                          reg_targets]  # [batch_size,h*w]
        left_right_max_list = [torch.max(reg_target[..., 0], reg_target[..., 2]) for reg_target in reg_targets]
        top_bottom_min_list = [torch.min(reg_target[..., 1], reg_target[..., 3]) for reg_target in reg_targets]
        top_bottom_max_list = [torch.max(reg_target[..., 1], reg_target[..., 3]) for reg_target in reg_targets]
        cnt_targets = [
            ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(dim=-1) for
            left_right_min, left_right_max, top_bottom_min, top_bottom_max in
            zip(left_right_min_list, left_right_max_list, top_bottom_min_list,
                top_bottom_max_list)]  # [batch_size,h*w,1]

        assert len(reg_targets) == batch_size and reg_targets[0].shape == (h_mul_w, 4)
        assert len(cls_targets) == batch_size and cls_targets[0].shape == (h_mul_w, 1)
        assert len(cnt_targets) == batch_size and cnt_targets[0].shape == (h_mul_w, 1)

        # process neg coords
        mask_pos_2 = [mask_p.long().sum(dim=-1) for mask_p in mask_pos]  # [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = [mask_p_2 >= 1 for mask_p_2 in mask_pos_2]
        assert len(mask_pos_2) == batch_size and mask_pos_2[0].shape[0] == (h_mul_w)
        for cls_target, cnt_target, reg_target, mask_p_2 in zip(cls_targets, cnt_targets, reg_targets, mask_pos_2):
            if cls_target.shape[0]>0:
                cls_target[~mask_p_2] = 0  # [batch_size,h*w,1]
            if cnt_target.shape[0] > 0:
                cnt_target[~mask_p_2] = -1 # [batch_size,h*w,1]
            if reg_target.shape[0] > 0:
                reg_target[~mask_p_2] = -1 # [batch_size,h*w,4]

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


class DetectHead2(nn.Module):
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
            # torch.gt(a, 0.2))
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