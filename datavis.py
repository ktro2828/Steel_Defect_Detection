#!/usr/bin/env python


import os.path as osp
import random

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.mask_utils import rle2mask


class DataVisualizer(object):
    """This is data visualizer"""

    def __init__(self):
        self.root = './dataset/'
        self.df_path = osp.join(self.root, 'train.csv')
        self.train_path = osp.join(self.root, 'train_images/')
        self.df = pd.read_csv(self.df_path)
        self.columns = 1
        self.rows = 4

    def datasize(self):
        print('NUM DATA: {}'.format(self.df.shape[0]))

    def display(self):
        fig = plt.figure(figsize=(20, 20))
        df = self.df[self.df['EncodedPixels'].notnull()]
        df = df.pivot(index='ImageId',
                      columns='ClassId',
                      values='EncodedPixels')
        for i in range(1, self.columns * self.rows + 1):
            idx = random.randint(0, len(self.df) - 1)
            fig.add_subplot(self.rows, self.columns, i)

            image_id, mask = rle2mask(idx, df)
            img_path = osp.join(self.train_path, image_id)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for ch in range(mask.shape[-1]):
                if mask[:, :, ch].any():
                    idx = ch
            mask = mask[:, :, idx]
            img[mask == 1, 0] = 255
            plt.imshow(img)
            plt.axis('off')
        plt.show()
        plt.close()


def main():
    visualizer = DataVisualizer()
    visualizer.datasize()
    visualizer.display()


if __name__ == '__main__':
    main()
