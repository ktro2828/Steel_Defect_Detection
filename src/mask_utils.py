#!/usr/bin/env python


import numpy as np


def make_mask(row_idx, df):
    """Given a row index, return image_id and mask (256, 1600, 4)
        from the data frame

    Args:
        row_idx(int): index of target image
        df(dataframe)
    """
    image_id = df.iloc[row_idx].name
    labels = df.iloc[row_idx][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')

    return image_id, masks


def mask2rle(img):
    """
    img: numpy array, 1 -> mask, 0 -> background
    Returns run length as string format
    """
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)
