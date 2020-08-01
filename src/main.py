#!/usr/bin/env python


import os
import os.path as osp
import warnings

import numpy as np
import torch

from args import arg_parser
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

    parser = arg_parser()
    args = parser.parse_args()

    if osp.exists('../trained_models') is False:
        os.makedirs('../trained_models')
    if osp.exists('../predictions') is False:
        os.makedirs('../predictions')

    if args.model == 'unet':
        model = UNet(3, 4)
    elif args.model == 'resunet_a':
        model = ResUNet_a(3, 4)
    else:
        model = smp.Unet(args.model, encoder_weights='imagenet',
                         classes=4, activation='None')

    model_trainer = Trainer(model, loss=args.loss)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    if osp.exists('../results') is False:
        os.makedirs('../results')
    plot(losses, name='{}_Loss'.format(args.loss))
    plot(dice_scores, name='Dice_score')
    plot(iou_scores, name='IoU_score')


if __name__ == '__main__':
    main()
