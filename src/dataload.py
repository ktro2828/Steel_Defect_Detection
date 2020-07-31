#!/usr/bin/env python


import os

from albumentations import HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise
from albumentations.pytorch import ToTensor
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from mask_utils import make_mask


class SteelDataManager(Dataset):
    """Steel Defects dataset"""

    def __init__(self, df, dir_path, mean, std, phase):
        """
        Args:
            df(dataframe): Dataframe
            dir_path(string): Directory with all the images
            mean(list): Mean value for augment
            std(list): Std Value for augment
            Phase(string): Phase train or test
        """
        super(SteelDataManager, self).__init__()
        self.df = df
        self.root = dir_path
        self.phase = phase
        self.mean = mean
        self.std = std
        self.fnames = self.df.index.tolist()
        self.transforms = get_transforms(phase, mean, std)

    def __getitem__(self, idx):
        image_id, mask = make_mask(idx, self.df)
        image_path = os.path.join(self.root, "train_images", image_id)
        image = io.imread(image_path)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)

        sample = {'image': image, 'mask': mask}

        return sample

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    tf_list = []
    if phase == 'train':
        tf_list.extend([HorizontalFlip(p=0.5)])
    tf_list.extend([
        Normalize(mean=mean, std=std, p=1),
        ToTensor()
    ])
    trans = Compose(tf_list)
    return trans


def dataloader(data_folder,
               df_path,
               phase,
               mean=None,
               std=None,
               batch_size=8,
               num_workers=4):
    df = pd.read_csv(df_path)
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df['defects'], random_state=1234)
    df = train_df if phase == 'train' else val_df
    image_dataset = SteelDataManager(df, data_folder, mean, std, phase)
    dl = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True
    )

    return dl
