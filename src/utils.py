#!/usr/bin/env python

import os
import os.path as osp
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


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


def _compute_ious(pred,
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


def _compute_iou_batch(outputs, labels, classes=None):
    """Compute mean iou for a batch of ground truth masks and predicted masks
    """

    ious = []
    preds = np.copy(outputs)
    labels = np.array(labels)
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(_compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)

    return iou


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
        outputs = torch.sigmoid(outputs)
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
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" %
          (epoch_loss, iou, dice, dice_neg, dice_pos))

    return dice, iou


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores['train'])),
             scores['train'], label='train{}'.format(name))
    plt.plot(range(len(scores['val'])),
             scores['val'], label='val {}'.format(name))
    plt.title('{}plot'.format(name))
    plt.xlabel('Epoch')
    plt.ylabel('{}'.format(name))
    plt.legend()
    plt.show()
    plt.savefig('../results/{}.png'.format(name))
    plt.close()


def visualize(sample, outputs, epoch, phase):
    batch_size = len(sample)
    idx = random.randint(0, batch_size - 1)
    images = sample['image'].cpu().detach().numpy()
    masks = sample['mask'].cpu().detach().numpy()

    images = np.transpose(images, (0, 2, 3, 1))[idx]

    masks = np.transpose(masks, (0, 2, 3, 1))[idx]
    for ch in range(masks.shape[-1]):
        if masks[:, :, ch].any():
            ch_idx = ch
    masks = masks[:, :, ch_idx]

    ground_truth = images.copy()
    ground_truth[masks == 1, 0] = 255

    predict = images.copy()
    outputs = torch.sigmoid(outputs)
    outputs = outputs.cpu().detach().numpy()
    outputs = np.transpose(outputs, (0, 2, 3, 1))[idx]
    thresh = np.mean(outputs) * 1.2
    outputs = outputs[:, :, ch_idx]
    outputs[outputs < thresh] = 0
    outputs[outputs > thresh] = 1
    predict[outputs == 1, 0] = 255

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))

    ax1.imshow(ground_truth)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    ax2.imshow(predict)
    ax2.set_title('Prediction')
    ax2.axis('off')

    plt.show()
    plt.savefig('../predictions/{}_{}.png'.format(phase, epoch))
    plt.close()
    print('******Saving {} image******'.format(phase))
