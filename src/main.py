#!/usr/bin/env python


import os
import os.path as osp

import numpy as np
import torch

from visualize import plot
from trainer import Trainer
from unet import UNet


def main():
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if osp.exists('../trained_models') is False:
        os.makedirs('../trained_models')

    model = UNet(4, 4)
    model_trainer = Trainer(model)
    model_trainer.start()

    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores
    iou_scores = model_trainer.iou_scores

    plot(losses, name='BCE loss')
    plot(dice_scores, name='Dice score')
    plot(iou_scores, 'IoU score')


if __name__ == '__main__':
    main()
