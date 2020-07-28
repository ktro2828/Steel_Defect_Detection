#!/usr/bin/env python


import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNet_a(nn.Module):
    """ResUNet-a

    Args:
        n_channels(int): channles input
        n_classes(int): output classes
    """

    def __init__(self, n_channel, n_classes):
        super(ResUNet_a, self).__init__()
        self.conv1 = nn.Conv2d(
            n_channel, 32, kernel_size=1, dilation=1, stride=1)
        self.resblock1 = ResBlock_a(32, 64, dilations=(1, 3, 15, 31))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.resblock2 = ResBlock_a(128, 256, dilations=(1, 3, 15))

    def forward(self, x):

        return


class Combine(nn.Module):
    """Combine block

    Args:
        in_ch(int): input channels
        out_ch(int): output channles
    """

    def __init__(self, in_ch, out_ch):
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x1 = F.relu(x1, inplace=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class ResBlock_a(nn.Module):
    """ResBlock-a

    Args:
        in_ch(int): number of input channels
        out_ch(int): number of output channels
        dilations(int, list or tuple): list of dilations
    """

    def __init__(self, in_ch, out_ch, dilations):
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
    """Basic block for ResBlock-a

    Args:
        in_ch(int): number of input channel
        out_ch(int): number of output channel
        d(int): dilation rate
    """

    def __init__(self, in_ch, out_ch, d):
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
    """PSPPool

    Args:
        in_ch(int): number of input channels
        out_ch(int): number of output channels
        portion(int): scale factor for PSP
    """

    def __init__(self, in_ch, out_ch, portion):
        super(PSPPool, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 2048, kernel_size=1, stride=2),
        self.block = ResBlock_a(2048, 2048, 1),
        self.pool = nn.Maxpool(kernel_size=2, stride=2)
        self.up = nn.Upsample((28, 28))
        self.conv2 = nn.Conv2d(out_ch, 2048, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.block(x)
        x2 = self.pool(x1)
        x2 = self.up(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv2(x)
        return x
