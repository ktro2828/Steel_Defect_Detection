#!/usr/bin/env python


import os
import os.path as osp
import warnings

import argparse
import numpy as np
import torch

from utils import plot
from trainer import Trainer
from unet import UNet
from resunet_a import ResUNet_a

import segmentation_models_pytorch as smp


def main():
    warnings.filterwarnings('ignore')
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loss', type=str, default='BCE',
                        help='BCE, Dice, BCEDice, IoU, Tanimoto')
    parser.add_argument('-m', '--model', type=str,
                        default='unet', help='UNet or ResUNet-a')

    args = parser.parse_args()

    if osp.exists('../trained_models') is False:
        os.makedirs('../trained_models')

    if args.model == 'unet':
        model = UNet(3, 4)
    elif args.model == 'resnet':
        model = smp.Unet('resnet18', encoder_weights='imagenet',
                         classes=4, activation='None')
    else:
        model = ResUNet_a(3, 4)

    model_trainer = Trainer(model, loss=args.loss)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    plot(losses, name='{}_Loss'.format(args.loss))
    plot(dice_scores, name='Dice_score')
    plot(iou_scores, name='IoU_score')


if __name__ == '__main__':
    main()
