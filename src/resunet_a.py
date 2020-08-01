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
        self.psppool1 = PSPPool(1024, outsize=(8, 50))
        self.up1 = Upsampling(512)
        self.comb1 = Combine(512)
        self.up_block1 = ResBlock_a(512, dilations=1)
        self.up2 = Upsampling(256)
        self.comb2 = Combine(256)
        self.up_block2 = ResBlock_a(256, dilations=1)
        self.up3 = Upsampling(128)
        self.comb3 = Combine(128)
        self.up_block3 = ResBlock_a(128, dilations=1)
        self.up4 = Upsampling(64)
        self.comb4 = Combine(64)
        self.up_block4 = ResBlock_a(64, dilations=1)
        self.up5 = Upsampling(32)
        self.comb5 = Combine(32)
        self.up_block5 = ResBlock_a(32, dilations=1)
        self.comb6 = Combine(32)
        self.psppool2 = PSPPool(32)
        self.conv_out = nn.Conv2d(32, n_classes, kernel_size=1, stride=1)

    def forward(self, x):
        c1 = x = self.conv1(x)
        c2 = x = self.block1(x)
        x = self.conv2(x)
        c3 = x = self.block2(x)
        x = self.conv3(x)
        c4 = x = self.block3(x)
        x = self.conv4(x)
        c5 = x = self.block4(x)
        x = self.conv5(x)
        c6 = x = self.block5(x)
        x = self.conv6(x)
        x = self.block6(x)
        x = self.psppool1(x)
        x = self.up1(x)
        x = self.comb1(x, c6)
        x = self.up_block1(x)
        x = self.up2(x)
        x = self.comb2(x, c5)
        x = self.up_block2(x)
        x = self.up3(x)
        x = self.comb3(x, c4)
        x = self.up_block3(x)
        x = self.up4(x)
        x = self.comb4(x, c3)
        x = self.up_block4(x)
        x = self.up5(x)
        x = self.comb5(x, c2)
        x = self.up_block5(x)
        x = self.comb6(x, c1)
        x = self.psppool2(x)
        x = self.conv_out(x)

        return x


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

    def __init__(self, out_ch, in_ch=None, scales=(1, 2, 4, 8), outsize=None):
        super(PSPPool, self).__init__()
        if in_ch is None:
            in_ch = out_ch
        if isinstance(scales, int):
            scales = tuple((scales,))
        self.scales = scales
        self.psppool = nn.ModuleList([
            self._build(in_ch, out_ch, s, outsize) for s in self.scales
        ])
        self.conv = nn.Conv2d(2*in_ch, out_ch, kernel_size=1, stride=1)
        self.shortcut = self._shortcut(in_ch, out_ch)

    def forward(self, x):
        output = []
        output.append(self.shortcut(x))
        for block in self.psppool:
            output.append(block(x))
        x = torch.cat(output, dim=1)
        x = self.conv(x)
        return x

    def _build(self, in_ch, out_ch, s, outsize):
        out_ch = int(out_ch / len(self.scales))
        return PSPBlock(in_ch, out_ch, s, outsize)

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

    def __init__(self, in_ch, out_ch, s, outsize=None):
        super(PSPBlock, self).__init__()
        if outsize is None:
            scale = s
        else:
            scale = None
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=s),
            nn.Conv2d(in_ch, out_ch, kernel_size=1),
            nn.Upsample(size=outsize, scale_factor=scale),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Upsampling(nn.Module):
    """Upsampling block

    Args:
        out_ch(int): number of output channels
        in_ch(int: default=None): number of input channels
    """

    def __init__(self, out_ch, in_ch=None, bilinear=True):
        super(Upsampling, self).__init__()
        if in_ch is None:
            in_ch = int(2 * out_ch)
        if bilinear:
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1),
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True)
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_ch / 2, out_ch / 2, kernel_size=2, stride=2
            )

    def forward(self, x):
        x = self.up(x)
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
        # x1.size() < x2.size()
        x1 = self.relu(x1)
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (int(diff_h / 2), int(diff_h / 2),
                        int(diff_w / 2), int(diff_w / 2)))
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x
