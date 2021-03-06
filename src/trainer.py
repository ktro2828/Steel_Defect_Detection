#!/usr/bin/env python


import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import line_notify

from dataload import dataloader
from utils import Meter, epoch_log, visualize
from lossfuncs import DiceLoss, DiceBCELoss, IoULoss, FocalLoss, TanimotoLoss


class Trainer(object):
    """Trainer class taking care of training and validation"""

    def __init__(self, model, loss='DiceBCE'):
        df_path = '../dataset/train.csv'
        root = osp.dirname(df_path)
        self.num_workers = 6
        self.batch_size = {'train': 3, 'val': 3}
        self.accumlation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 40
        self.phases = ['train', 'val']
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = model.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, verbose=True)
        if loss == 'BCE':
            self.criterion = nn.BCELoss()
        elif loss == 'Dice':
            self.criterion = DiceLoss()
        elif loss == 'DiceBCE':
            self.criterion = DiceBCELoss()
        elif loss == 'IoU':
            self.criterion = IoULoss()
        elif loss == 'Focal':
            self.criterion = FocalLoss()
        elif loss == 'Tanimoto':
            self.criterion = TanimotoLoss()
        else:
            raise ValueError('Loss Function is not Defined')
            return

        self.dataloaders = {
            phase: dataloader(
                root=root,
                df_path=df_path,
                phase=phase,
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers
            )
            for phase in self.phases
        }
        self.best_loss = float('inf')
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.iou_scores = {phase: [] for phase in self.phases}

        self.client = line_notify.LineNotify(
            token='buNeQjYHp6sXPdwk1sMWUCmqLQr7z7czjLozKJdtevL'
        )

    def _forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def _iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime('%H:%M:%S')
        print("Starting epoch: {} | phase: {} | Time: {}".format(
            epoch + 1, phase, start))
        dl = self.dataloaders[phase]
        running_loss = 0.0
        total_steps = len(dl)
        self.optimizer.zero_grad()
        for itr, sample in enumerate(tqdm(dl)):
            images = sample['image']
            targets = sample['mask']
            loss, outputs = self._forward(images, targets)
            loss /= self.accumlation_steps
            if phase == 'train':
                loss.backward()
                if (itr + 1) % self.accumlation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        epoch_loss = (running_loss * self.accumlation_steps) / total_steps
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        visualize(sample, outputs, epoch, phase)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)

        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self._iterate(epoch, 'train')
            state = {
                'epoch': epoch,
                'best_loss': self.best_loss,
                'state_dict': self.net.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }
            with torch.no_grad():
                val_loss = self._iterate(epoch, 'val')
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print('******** New optimal found, saving state ********')
                state['best_loss'] = self.best_loss = val_loss
                torch.save(state, "../trained_models/model.pth")

            self.client.notify('Epoch: {} Done!'.format(epoch))
