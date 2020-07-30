#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np
import torch

from lossfuncs import DiceLoss, DiceBCELoss, IoULoss, TanimotoLoss


def _metric(prob, truth, threshold, reduction=None):
    """Calculate dice of positive and negative images separately

    Args:
        prob(torch.tensors): output of model
        truth(torch.tensors): ground truth
    """
    batch_size = len(truth)
    with torch.no_grad():
        prob = prob.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(prob.shape == truth.shape)

        p = (prob > threshold).float()
        t = (truth > threshold).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / (p + t).sum(-1)

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

        num_neg = len(neg_index)
        num_pos = len(pos_index)

    ret = {
        'dice': dice, 'dice_neg': dice_neg, 'dice_pos': dice_pos,
        'num_neg': num_neg, 'num_pos': num_pos
    }

    return ret


def _predict(x, threshold):
    x_p = np.copy(x)
    preds = (x_p > threshold).astype('uint8')
    return preds


def _compute_iou_batch(pred,
                       label,
                       classes,
                       ignore_index=255,
                       only_present=True):
    """Compute IoU for each batch

    Args:
        pred(nd-array):
        label(nd-array):
        classes(list):
        ignore_index(int; default=255):
        only_present(bool; default=True):
    """
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]


class Meter(object):
    """A Meter to keep track of IoU and Dice scores throughout an eopch"""

    def __init__(self, phase, epoch):
        self.threshold = 0.5

        self.scores = {
            'base_dice': [],
            'dice_neg': [],
            'dice_pos': [],
            'iou': []
        }

    def update(self, targets, outputs):
        ret = _metric(outputs, targets, self.threshold)
        self.scores['base_dice'].extend(ret['dice'].tolist())
        self.scores['dice_pos'].extend(ret['dice_pos'].tolist())
        self.scores['dice_neg'].extend(ret['dice_neg'].tolist())
        preds = _predict(outputs, self.threshold)
        iou = _compute_iou_batch(preds, targets, classes=[1])
        self.scores['iou'].append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.scores['base_dice'])
        dice_neg = np.nanmean(self.scores['dice_neg'])
        dice_pos = np.nanmean(self.scores['dice_pos'])
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.scores['iou'])

        return dices, iou


def epoch_log(phase, epoch, epoch_loss, meter, start):
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" %
          (epoch_loss, dice, dice_neg, dice_pos))

    return dice, iou


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores['train'])),
             scores['train'], label='train{}'.format(name))
    plt.title('{}plot'.format(name))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(name))
    plt.legend()
    plt.show()
    plt.savefig('{}.png'.format(name))
    plt.close()
