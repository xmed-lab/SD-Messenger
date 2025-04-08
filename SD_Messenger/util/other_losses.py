# -*- coding: utf-8 -*-
# @Time    : 2023/10/28 22:00
# @Author  : Haonan Wang
# @File    : other_losses.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F

from loss import SoftDiceLoss


def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight


class Difficulty:
    def __init__(self, num_cls, accumulate_iters=20):
        self.last_dice = torch.zeros(num_cls).float().cuda() + 1e-8
        self.dice_func = SoftDiceLoss(smooth=1e-8, do_bg=True)
        self.cls_learn = torch.zeros(num_cls).float().cuda()
        self.cls_unlearn = torch.zeros(num_cls).float().cuda()
        self.num_cls = num_cls
        self.dice_weight = torch.ones(num_cls).float().cuda()
        self.accumulate_iters = accumulate_iters

    def init_weights(self):
        weights = np.ones(self.num_cls) * self.num_cls
        self.weights = torch.FloatTensor(weights).cuda()
        return weights

    def cal_weights(self, pred, label):
        x_onehot = torch.zeros(pred.shape).cuda()
        output = torch.argmax(pred, dim=1, keepdim=True).long()
        x_onehot.scatter_(1, output, 1)
        y_onehot = torch.zeros(pred.shape).cuda()
        y_onehot.scatter_(1, label, 1)
        cur_dice = self.dice_func(x_onehot, y_onehot, is_training=False)
        delta_dice = cur_dice - self.last_dice
        cur_cls_learn = torch.where(delta_dice > 0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)
        cur_cls_unlearn = torch.where(delta_dice <= 0, delta_dice, 0) * torch.log(cur_dice / self.last_dice)

        self.last_dice = cur_dice

        self.cls_learn = EMA(cur_cls_learn, self.cls_learn,
                             momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        self.cls_unlearn = EMA(cur_cls_unlearn, self.cls_unlearn,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        cur_diff = (self.cls_unlearn + 1e-8) / (self.cls_learn + 1e-8)

        cur_diff = torch.pow(cur_diff, 1 / 5)

        self.dice_weight = EMA(1. - cur_dice, self.dice_weight,
                               momentum=(self.accumulate_iters - 1) / self.accumulate_iters)
        weights = cur_diff * self.dice_weight
        # weights = weights / weights.max()
        return weights * self.num_cls


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """

    def __init__(self, weight=None):
        if weight is not None:
            weight = torch.FloatTensor(weight).cuda()
        super().__init__(weight=weight)

    def forward(self, input, target):
        return super().forward(input, target)

    def update_weight(self, weight):
        self.weight = weight


class MultiConsistLoss(nn.Module):
    def __init__(self):
        super(MultiConsistLoss, self).__init__()
        self._criterion1 = nn.MSELoss()
        self._criterion2 = nn.MSELoss()
        self._criterion3 = nn.MSELoss()
        self._criterion4 = nn.MSELoss()

    def forward(self, features):
        f1 = features[0]
        f2 = features[1]
        f3 = features[2]
        f4 = features[3]
        loss1 = self._criterion1(f1, f2)
        loss2 = self._criterion2(f2, f3)
        loss3 = self._criterion3(f3, f4)
        loss4 = self._criterion4(f4, f1)
        loss = (loss1 + loss2 + loss3 + loss4) / 4.0
        return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        if weight.dim() > 1:
            assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


class UncertaintyCELoss(nn.Module):
    def __init__(self, ignore_index):
        super(UncertaintyCELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred_u, pseudo_label):
        # print(f'pred shape: {pred_u.shape}')
        # print(f'label shape: {pseudo_label.shape}')
        loss = F.cross_entropy(pred_u, pseudo_label, ignore_index=self.ignore_index, reduction='none')
        # weight = torch.ones_like(loss)
        # print(f'loss shape: {loss}')
        metric = -loss.detach().reshape((loss.shape[0], loss.shape[1] * loss.shape[2]))
        # print(f'metric shape: {torch.max(metric, dim=1)}')
        weight = F.softmax(metric, 1) * metric.size(1)
        # print(f'weight: {weight}')
        # weight = weight / weight.mean(1).reshape((-1, 1))
        # print(f'weight: {weight}')
        weight = weight.reshape((loss.shape[0], loss.shape[1], loss.shape[2]))

        # print(weight)

        for i in range(pseudo_label.shape[0]):
            tag = set(pseudo_label[i, :, :].reshape(pseudo_label.shape[1] * pseudo_label.shape[2]).tolist()) - {255}
            if len(tag) <= 1:
                weight[i] = 1
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(loss, weight=weight)
        return loss
