#!/usr/bin/env python

"""
albumentations:
    ShiftScaleRotate, Resize, GaussNoise
"""

import os.path as osp

from albumentations import HorizontalFlip, Normalize, Compose
from albumentations.pytorch import ToTensor
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from mask_utils import rle2mask


class SteelDataManager(Dataset):
    """Steel Defects dataset"""

    def __init__(self, root, df, phase):
        """
        Args:
            root(string): Root path of dataset directory
            df(dataframe): Dataframe
            mean(list): Mean value for augment
            std(list): Std Value for augment
            Phase(string): Phase train or test
        """
        super(SteelDataManager, self).__init__()
        self.root = root
        self.df = df
        self.phase = phase

        if self.phase == 'train':
            self.data_path = osp.join(self.root, 'train_images/')
        if self.phase == 'test':
            self.data_path = osp.join(self.root, 'test_images/')
        self.transforms = get_transforms(phase)

    def __getitem__(self, idx):
        image_id, mask = rle2mask(idx, self.df)
        image_path = osp.join(self.data_path, image_id)
        image = io.imread(image_path)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)

        ret = {'image': image, 'mask': mask}

        return ret

    def __len__(self):
        return len(self.df)


def get_transforms(phase):
    tf_list = []
    if phase == 'train':
        tf_list.extend([HorizontalFlip(p=0.5)])
    tf_list.extend([
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225), p=1),
        ToTensor()
    ])
    trans = Compose(tf_list)
    return trans


def dataloader(root,
               df_path,
               phase,
               batch_size=8,
               num_workers=4):
    df = pd.read_csv(df_path)
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['defects'], random_state=1234)
    df = train_df if phase == 'train' else val_df
    image_dataset = SteelDataManager(root, df, phase)
    dl = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )

    return dl
