#!/usr/bin/env python

"""inputs = F.sigmoid(inputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE

        return focal_loss


class TanimotoLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TanimotoLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = F.softmax(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        square_i = torch.square(inputs)
        square_t = torch.square(targets)

        sum_product = (inputs * targets).sum()
        denominator = (square_i + square_t).sum() - sum_product
        tanimoto = (sum_product + smooth) / (denominator + smooth)

        return 1 - tanimoto


class TanimotoDualLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TanimotoDualLoss, self).__init__()
        self.tanimoto = TanimotoLoss()

    def forward(self, inputs, targets, smooth=1):
        t_1 = self.tanimoto(inputs, targets)            # 1 - T(p, l)
        t_2 = self.tanimoto(1 - inputs, 1 - targets)    # 1 - T(1-p, 1-l)
        tanimoto = (t_1 + t_2) / 2

        return tanimoto
