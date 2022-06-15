# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2021/2/3 9:35
# @Author : liumin
# @File : seg_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .lovasz_losses import lovasz_softmax


def make_one_hot(labels, classes, ignore_index=None):
    labels = labels.unsqueeze(dim=1)
    if ignore_index is not None:
        ignore_i = labels == ignore_index
        # preset the element to be ignored to 0
        labels[ignore_i] = 0
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
        # remove ignore index
        ignore_i = ignore_i.expand(ignore_i.shape[0], classes, ignore_i.shape[2], ignore_i.shape[3])
        target[ignore_i] = 0
    else:
        one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
        target = one_hot.scatter_(1, labels.data, 1)
    return target


class NLLLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.nll = nn.NLLLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[1:]
        return self.nll(output, target.long())


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=255, reduction='mean'):
        super(CrossEntropyLoss2d, self).__init__()
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[1:]
        return self.CE(output, target.long())


class BootstrappedCELoss2d(nn.Module):
    def __init__(self, min_K, threshold=0.3, weight=None, ignore_index=255, reduction='none'):
        super(BootstrappedCELoss2d, self).__init__()
        self.min_K = min_K  # minK = int(batch_size * h * w / 16)
        self.threshold = threshold
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        pixel_losses = self.criterion(output, target.long()).contiguous().view(-1)

        mask = (pixel_losses > self.threshold)
        if torch.sum(mask).item()>self.min_K:
            pixel_losses=pixel_losses[mask]
        else:
            pixel_losses, _ = torch.topk(pixel_losses, self.min_K)
        return pixel_losses.mean()

'''
class OhemCrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, thresh=0.7, min_kept=100000, ignore_index=255, reduction='mean'):
        super(OhemCrossEntropyLoss2d, self).__init__()
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.ignore_index = ignore_index
        self.CE = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[1:]
        target = target.squeeze(1).long()

        # ohem
        n, c, h, w = output.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(output, dim=1)
        prob = prob.transpose(0, 1).reshape(c, -1)

        if num_valid > 0:
            # prob = prob.masked_fill_(1 - valid_mask, 1)
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
            kept_mask = mask_prob.le(threshold)
            valid_mask = valid_mask * kept_mask
            target = target * kept_mask.long()

        # target = target.masked_fill_(1 - valid_mask, self.ignore_index)
        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(n, h, w)
        return self.CE(output, target)
'''


class OhemCrossEntropyLoss(nn.Module):
    """
    Implements the ohem cross entropy loss function.
    Args:
        thresh (float, optional): The threshold of ohem. Default: 0.7.
        min_kept (int, optional): The min number to keep in loss computation. Default: 10000.
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, thresh=0.7, min_kept=10000, ignore_index=255):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thresh
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.EPS = 1e-5

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C), where C is number of classes, and if shape is more than 2D, this
                is (N, C, D1, D2,..., Dk), k >= 1.
            label (Tensor): Label tensor, the data type is int64. Shape is (N), where each
                value is 0 <= label[i] <= C-1, and if shape is more than 2D, this is
                (N, D1, D2,..., Dk), k >= 1.
        """
        if len(label.shape) != len(logit.shape):
            label = torch.unsqueeze(label, 1)

        # get the label after ohem
        n, c, h, w = logit.shape
        label = label.reshape((-1, ))
        valid_mask = (label != self.ignore_index).astype('int64')
        num_valid = valid_mask.sum()
        label = label * valid_mask

        prob = F.softmax(logit, dim=1)
        prob = prob.transpose(0, 1).reshape((c, -1))

        if self.min_kept < num_valid and num_valid > 0:
            # let the value which ignored greater than 1
            prob = prob + (1 - valid_mask)

            # get the prob of relevant label
            label_onehot = F.one_hot(label, c)
            label_onehot = label_onehot.transpose((1, 0))
            prob = prob * label_onehot
            prob = torch.sum(prob, dim=0)

            threshold = self.thresh
            if self.min_kept > 0:
                index = prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                threshold_index = int(threshold_index.numpy()[0])
                if prob[threshold_index] > self.thresh:
                    threshold = prob[threshold_index]
                kept_mask = (prob < threshold).astype('int64')
                label = label * kept_mask
                valid_mask = valid_mask * kept_mask

        # make the invalid region as ignore
        label = label + (1 - valid_mask) * self.ignore_index

        label = label.reshape((n, 1, h, w))
        valid_mask = valid_mask.reshape((n, 1, h, w)).astype('float32')
        loss = F.cross_entropy(F.softmax(logit, dim=1), label, ignore_index=self.ignore_index)
        # loss = F.softmax_with_cross_entropy(logit, label, ignore_index=self.ignore_index, axis=1)
        loss = loss * valid_mask
        avg_loss = torch.mean(loss) / (torch.mean(valid_mask) + self.EPS)

        label.stop_gradient = True
        valid_mask.stop_gradient = True
        return avg_loss


class OhemCrossEntropyLoss2d(nn.Module):
    def __init__(self, thresh=0.7, min_kept=100000, ignore_index=255, *args, **kwargs):
        super(OhemCrossEntropyLoss2d, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels.long()).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)



class BCEWithLogitsLoss2d(nn.Module):
    def __init__(self, weight=None, pos_weight=None, ignore_index=255, reduction='mean'):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[1:]
        target = make_one_hot(target.long(), output.size()[1], self.ignore_index)
        weight = None
        if self.weight is not None:
            if self.weight.shape != target.shape:
                weight = self.weight.view(1, -1, 1, 1).expand(target.shape).cuda()
            else:
                weight = self.weight.cuda()
        pos_weight = None
        if self.pos_weight is not None:
            if self.pos_weight.shape != target.shape:
                pos_weight = self.pos_weight.view(1, -1, 1, 1).expand(target.shape).cuda()
            else:
                pos_weight = self.pos_weight.cuda()
        loss = F.binary_cross_entropy_with_logits(output, target.float(), weight=weight, pos_weight=pos_weight,
                                                 reduction=self.reduction)
        return loss / output.shape[0]


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., weight=None, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.weight = weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, output, target):
        assert output.shape[0] == target.shape[0] and output.shape[2:] == target.shape[1:]
        target = make_one_hot(target.long(), output.size()[1], self.ignore_index)
        output = self.softmax(output)
        target = target.float()

        loss = 0
        for i in range(output.shape[1]):
            output_flat = output[:, i, :, :].contiguous().view(-1)
            target_flat = target[:, i, :, :].contiguous().view(-1)
            intersection = (output_flat * target_flat).sum()
            dice_loss = 1 - ((2. * intersection + self.smooth) /
                        (output_flat.sum() + target_flat.sum() + self.smooth))
            if self.weight is not None:
                dice_loss *= self.weight[i]
            loss += dice_loss
        return loss / output.shape[1]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5, weight=None, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.CE = CrossEntropyLoss2d(weight=self.weight,ignore_index=ignore_index, reduction=reduction)

    def forward(self, output, target):
        if output.dim()>2:
            output = output.contiguous().view(output.size(0), output.size(1), -1)
            output = output.transpose(1,2)
            output = output.contiguous().view(-1, output.size(2)).squeeze()
        if target.dim()==4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1,2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim()==3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        logpt = self.CE(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt) ** self.gamma) * self.alpha * logpt
        return loss


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index

    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, classes=self.classes, ignore=self.ignore_index)
        return loss


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, weight=None, ignore_index=255, reduction='mean'):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.ce = CrossEntropyLoss2d(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        ce_loss = self.ce(output, target)
        dice_loss = self.dice(output, target)
        return ce_loss + dice_loss


def dice_loss_func(input, target):
    smooth = 1.
    n = input.size(0)
    iflat = input.view(n, -1)
    tflat = target.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class DetailAggregateLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DetailAggregateLoss, self).__init__()

        self.laplacian_kernel = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32).reshape(1, 1, 3, 3).requires_grad_(False).type(torch.cuda.FloatTensor)

        self.fuse_kernel = torch.nn.Parameter(torch.tensor([[6. / 10], [3. / 10], [1. / 10]],
                                                           dtype=torch.float32).reshape(1, 3, 1, 1).type(
            torch.cuda.FloatTensor))

    def forward(self, boundary_logits, gtmasks):
        # boundary_logits = boundary_logits.unsqueeze(1)
        boundary_targets = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel, padding=1)
        boundary_targets = boundary_targets.clamp(min=0)
        boundary_targets[boundary_targets > 0.1] = 1
        boundary_targets[boundary_targets <= 0.1] = 0

        boundary_targets_x2 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=2, padding=1)
        boundary_targets_x2 = boundary_targets_x2.clamp(min=0)

        boundary_targets_x4 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=4, padding=1)
        boundary_targets_x4 = boundary_targets_x4.clamp(min=0)

        boundary_targets_x8 = F.conv2d(gtmasks.unsqueeze(1).type(torch.cuda.FloatTensor), self.laplacian_kernel,
                                       stride=8, padding=1)
        boundary_targets_x8 = boundary_targets_x8.clamp(min=0)

        boundary_targets_x8_up = F.interpolate(boundary_targets_x8, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x4_up = F.interpolate(boundary_targets_x4, boundary_targets.shape[2:], mode='nearest')
        boundary_targets_x2_up = F.interpolate(boundary_targets_x2, boundary_targets.shape[2:], mode='nearest')

        boundary_targets_x2_up[boundary_targets_x2_up > 0.1] = 1
        boundary_targets_x2_up[boundary_targets_x2_up <= 0.1] = 0

        boundary_targets_x4_up[boundary_targets_x4_up > 0.1] = 1
        boundary_targets_x4_up[boundary_targets_x4_up <= 0.1] = 0

        boundary_targets_x8_up[boundary_targets_x8_up > 0.1] = 1
        boundary_targets_x8_up[boundary_targets_x8_up <= 0.1] = 0

        boudary_targets_pyramids = torch.stack((boundary_targets, boundary_targets_x2_up, boundary_targets_x4_up),
                                               dim=1)

        boudary_targets_pyramids = boudary_targets_pyramids.squeeze(2)
        boudary_targets_pyramid = F.conv2d(boudary_targets_pyramids, self.fuse_kernel)

        boudary_targets_pyramid[boudary_targets_pyramid > 0.1] = 1
        boudary_targets_pyramid[boudary_targets_pyramid <= 0.1] = 0

        if boundary_logits.shape[-1] != boundary_targets.shape[-1]:
            boundary_logits = F.interpolate(
                boundary_logits, boundary_targets.shape[2:], mode='bilinear', align_corners=True)

        bce_loss = F.binary_cross_entropy_with_logits(boundary_logits, boudary_targets_pyramid)
        dice_loss = dice_loss_func(torch.sigmoid(boundary_logits), boudary_targets_pyramid)
        return bce_loss, dice_loss



if __name__ == '__main__':
    input = torch.randn(4, 5, 6, 7)
    target = torch.randn(4, 5, 6, 7)

    ce_loss = CrossEntropyLoss2d()
    output = ce_loss(input, target)
    print('ce_loss', output.item())


    target = torch.ones([5, 6], dtype=torch.float32)  # 64 classes, batch size = 10
    output = torch.full([5, 6], 1.5)  # A prediction (logit)
    pos_weight = torch.ones([6])  # All weights are equal to 1
    criterion = BCEWithLogitsLoss2d(pos_weight=pos_weight)
    criterion(output, target)  # -log(sigmoid(1.5))
    print('bce_loss', output)
