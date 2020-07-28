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
        self.conv1 = nn.Conv2d(n_channel, 32, kernel_size=1, stride=1)
        self.block1 = ResBlock_a(32, dilations=(1, 3, 15, 31))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=2)
        self.block2 = ResBlock_a(64, dilations=(1, 3, 15, 31))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, stride=2)
        self.block3 = ResBlock_a(128, dilations=(1, 3, 15))
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, stride=2)
        self.block4 = ResBlock_a(256, dilations=(1, 3, 15))
        self.conv5 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        self.block5 = ResBlock_a(512, dilations=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)
        self.block6 = ResBlock_a(1024, dilations=1)
        self.psppool1 = PSPPool(1024)

    def forward(self, x):

        return


class ResBlock_a(nn.Module):
    """ResBlock-a

    Args:
        out_ch(int): number of output channels
        dilations(int, list[int] or tuple(int)): dilation rates
        in_ch(int: default=None): number of input channels
    """

    def __init__(self, out_ch, dilations, in_ch=None):
        super(ResBlock_a, self).__init__()
        if isinstance(dilations, int):
            dilations = tuple((dilations,))
        if in_ch is None:
            in_ch = out_ch

        self.resblock = nn.ModuleList([
            self._build(in_ch, out_ch, d) for d in dilations
        ])
        self.shortcut = self._shortcut(in_ch, out_ch)

    def forward(self, x):
        h = self.shortcut(x)
        for block in self.resblock:
            h += block(x)
        return h

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
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)


class Block(nn.Module):
    """Basic block for ResBlock-a

    Args:
        in_ch(int): number of input channels
        out_ch(int): number of output channels
        d(int): dilation rate
    """

    def __init__(self, in_ch, out_ch, d):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      stride=1, padding=d, dilation=d),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3,
                      stride=1, padding=d, dilation=d)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class PSPPool(nn.Module):
    """PSPPooling

    Args:
        out_ch(int): number of output channels
        in_ch(int: default=None): number of input channels
        scales(int, list[int] or tuple(int): default=(1, 2, 3, 4)):
            scale factors for PSPPooling
    """

    def __init__(self, out_ch, in_ch=None, scales=(1, 2, 4, 8)):
        super(PSPPool, self).__init__()
        if in_ch is None:
            in_ch = out_ch
        if isinstance(scales, int):
            scales = tuple((scales,))
        self.psppool = nn.ModuleList([
            self._build(in_ch, out_ch, p) for p in scales
        ])
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1)
        self.shortcut = self._shortcut(in_ch, out_ch)

    def forward(self, x):
        output = list(self.shortcut(x))
        for block in self.psppool:
            output.append(block(x))
        x = torch.cat(output, dim=1)
        x = self.conv(x)
        return x

    def _build(self, in_ch, s):
        out_ch = in_ch
        return PSPBlock(in_ch, out_ch, s)

    def _shortcut(self, in_ch, out_ch):
        if in_ch != out_ch:
            return self._projection(in_ch, out_ch)
        else:
            return lambda x: x

    def _projection(self, in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=1)


class PSPBlock(nn.Module):
    """Basic Block for PSPPool

    Args:
        in_ch(int): number of input channels
        out_ch(int): number of output channels
        s(int): scale factor for filter
    """

    def __init__(self, in_ch, out_ch, s):
        super(PSPBlock, self).__init__()
        out_ch = int(out_ch / 4)
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=s),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.Upsample(size=(8, 8)),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Combine(nn.Module):
    """Combine block

    Args:
        out_ch(int): number of output channles
        in_ch(int: default=None): number of input channels
    """

    def __init__(self, out_ch, in_ch=None):
        super(Combine, self).__init__()
        if in_ch is None:
            in_ch = out_ch
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(2 * in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x1 = self.relu(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        return x
