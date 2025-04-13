# -*- coding: utf-8 -*-
# @Time    : 2023/10/28 22:00
# @Author  : Haonan Wang
# @File    : other_losses.py
# @Software: PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F


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
