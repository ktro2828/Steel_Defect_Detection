#!/usr/bin/env python


import os
import os.path as osp

import argparse
import numpy as np
import torch

from utils import plot
from trainer import Trainer
from unet import UNet
from resunet_a import ResUNet_a


def main(args):
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loss', type=str, default='BCE',
                        help='BCE, Dice, BCEDice, IoU, Tanimoto')
    parser.add_argument('-m', '--model', type=str, default='unet', help='UNet or ResUNet-a')

    args = parser.parse_args()

    if osp.exists('../trained_models') is False:
        os.makedirs('../trained_models')

    if args.model == 'unet':
        model = UNet(3, 4)
    else:
        model = ResUNet_a(3, 4)
    model_trainer = Trainer(model, loss=args.loss)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    plot(losses, name='{} Loss'.format(args.loss))
    plot(dice_scores, name='Dice score')
    plot(iou_scores, name='IoU score')


if __name__ == '__main__':
    import sys
    argv = sys.argv
    main(argv)
