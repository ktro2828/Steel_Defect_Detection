#!/usr/bin/env python


import torch.nn as nn


class ResUNet_a(nn.Module):
    """ResUNet-a"""

    def __init__(self, n_channel):
        super(ResUNet_a, self).__init__()
        self.conv1 = nn.Conv2d(
            n_channel, 32, kernel_size=1, dilation=1, stride=1)

    def forward(self, input):

        return


class ResBlock_a(nn.Module):
    """ResBlock-a"""

    def __init__(self, in_ch, out_ch, dilations):
        """
        Args:
            in_ch(int): number of input channels
            out_ch(int): number of output channels
            dilations(int, list or tuple): list of dilations
        """
        super(ResBlock_a, self).__init__()
        if isinstance(int, dilations):
            dilations = tuple(dilations)

        self.resblock = nn.MoudleList([
            self._build(in_ch, out_ch, d) for d in dilations
        ])

    def forward(self, x):
        h = 0
        for block in self.resblock:
            h += block(x)
        x = self._shortcut(x)
        return h + x

    def _build(self, in_ch, out_ch, d):
        if in_ch is None:
            in_ch = out_ch
        return Block(in_ch, out_ch, d)

    def _shortcut(self, in_ch, out_ch):
        if in_ch != out_ch:
            return self._projection(in_ch, out_ch)
        else:
            return lambda x: x

    def _projection(self, in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)


class Block(nn.Module):
    """Basic block for ResBlock-a"""

    def __init__(self, in_ch, out_ch, d):
        """
        Args:
            in_ch(int): number of input channel
            out_ch(int): number of output channel
            d(int): dilation rate
        """
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, dilation=d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, dilation=d)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class PSPPool(nn.Module):
    """PSPPool"""

    def __init__(self, in_ch, out_ch):
        """
        Args:
            in_ch(int): number of input channel
            out_ch(int): number of output channel
        """
        super(PSPPool, self).__init__()
        self.pool = nn.ModuleList([
            self._build(in_ch, out_ch, portion) for portion in tuple(1, 2, 4, 8)
        ])

    def forward(self, x):
        h = 0
        for block in self.pool:
            h += block(x)
        x = self._shortcut(x)
        return h + x

    def _build(self, in_ch, out_ch, portion):
        if in_ch is None:
            in_ch = out_ch
        return Pool(in_ch, out_ch, portion)

    def _shortcut(self, in_ch, out_ch):
        if in_ch != out_ch:
            return self._projection(in_ch, out_ch)
        else:
            return lambda x: x

    def _projection(self, in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)


class Pool(nn.Module):
    """Basic block for PSPPool"""

    def __init__(self, in_ch, out_ch, portion):
        """
        Args:
            in_ch(int): number of input channels
            out_ch(int): number of output channels
            portion(int): scale factor for PSP
        """
        super(Pool, self).__init__()
        self.pool = nn.Sequential(
            nn.Conv2d(in_ch, 2048, kernel_size=1, stride=2),
            ResBlock_a(2048, 2048, 1),
            nn.Maxpool(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.pool(x)
        return x
