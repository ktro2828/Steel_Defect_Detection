#!/usr/bin/env python


import os
import os.path as osp

# import argparse
import numpy as np
import torch

from utils import plot
from trainer import Trainer
from unet import UNet


def main(args):
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-l', '--loss', type=str,
    #                     help='BCE, Dice, DiceBCE, IoU, Tanimoto')
    #
    # args = parser.parse_args()

    if osp.exists('../trained_models') is False:
        os.makedirs('../trained_models')

    model = UNet(3, 4)
    model_trainer = Trainer(model, loss=args[-1])
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    plot(losses, name='{} Loss'.format(args[-1]))
    plot(dice_scores, name='Dice score')
    plot(iou_scores, 'IoU score')


if __name__ == '__main__':
    import sys
    argv = sys.argv
    main(argv)
