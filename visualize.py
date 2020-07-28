#!/usr/bin/env python


from collections import defaultdict
import os.path as osp
from os import Path

import pandas as pd
from PIL import Image


class DataVisualizer(object):
    """This is data visualizer"""

    def __init__(self):
        self.root = Path('./dataset/')
        self.df_path = osp.join(self.root, 'train.csv')
        self.train_path = osp.join(self.root, 'train_images/')
        self.df = pd.read_csv(self.df_path)
        self.train_size_dict = defaultdict(int)
        self.classes = (1, 2, 3, 4)

    def _datasize(self):
        for image_name in self.train_path.iterdir():
            img = Image.open(image_name)
            self.train_size_dict[img.size] += 1
