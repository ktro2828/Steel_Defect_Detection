#!/usr/bin/env python


import os.path as osp

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
        df = self.df[self.df['EncodedPixels'].notnull()]
        print('NUM DATA: {}'.format(df.shape[0]))

    def display(self):
        print('==> Display Datas')
        fig = plt.figure(figsize=(20, 20))
        for i in range(1, self.columns*self.rows+1):
            fig.add_subplot(self.rows, self.columns, i)

            image_id, mask = rle2mask(i, self.df)
            img_path = osp.join(self.train_path, image_id)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img[mask == 1, 0] = 255
            plt.imshow(img)

        plt.show()
        plt.close()


def main():
    visualizer = DataVisualizer()
    visualizer.datasize()
    visualizer.imshow()


if __name__ == '__main__':
    main()
