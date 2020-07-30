#!/usr/bin/env python


import os.path as osp

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.mask_utils import make_mask


class DataVisualizer(object):
    """This is data visualizer"""

    def __init__(self):
        self.root = './dataset/'
        self.df_path = osp.join(self.root, 'train.csv')
        self.train_path = osp.join(self.root, 'train_images/')
        self.df = pd.read_csv(self.df_path)
        self.classes = (1, 2, 3, 4)

    def datasize(self):
        df = self.df[self.df['EncodedPixels'].notnull()]
        print('NUM DATAs: {}'.format(df.shape[0]))

    def imshow(self):
        fig = plt.figure(figsize=(20, 20))
        for i in range(1, 20 + 1):
            fig.add_subplot(10, 2, 1)

            image_id, mask = make_mask(i, self.df)
            path = osp.join(self.train_path, img_id)
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img[mask==1,0] = 255

            plt.imshow(img)

        plt.show()

def main():
    visualizer = DataVisualizer()
    visualizer.datasize()
    visualizer.imshow()


if __name__ == '__main__':
    main()
