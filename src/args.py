#!/usr/bin/env pyhton


import argparse


def arg_parser():
    """Argument Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loss', type=str, default='BCE',
                        help='BCE, Dice, BCEDice, IoU, Tanimoto')
    parser.add_argument('-m', '--model', type=str,
                        default='unet',
                        help='unet, resunet_a or resnet_x-based unet')

    return parser
