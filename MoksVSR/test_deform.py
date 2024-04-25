import os, glob, multiprocessing, torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import numpy as np
import erqa
import matplotlib.pyplot as plt
import time
import cv2
import torchvision.transforms as T
from torchvision.ops import deform_conv2d
import torch
import torch.nn as nn


def main():
    a = torch.randn(1, 128, 180, 320)
    out_channels = 64
    deform_groups = 16
    stride = 1
    padding = 1
    dilation = 1
    input_channels=128
    max_residue_magnitude = 10

    # conv_offset = nn.Sequential(
    #     nn.Conv2d(3 * out_channels + 4, self.out_channels, 3, 1, 1),
    #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #     nn.Conv2d(out_channels, self.out_channels, 3, 1, 1),
    #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #     nn.Conv2d(out_channels, self.out_channels, 3, 1, 1),
    #     nn.LeakyReLU(negative_slope=0.1, inplace=True),
    #     nn.Conv2d(out_channels, 27 * self.deform_groups, 3, 1, 1), # 3 * kernel3 * kernel3 * groups
    # )

    weight = nn.Parameter(
        torch.Tensor(out_channels, input_channels, 3, 3))
    #self.groups = 1                 
    #self.deform_groups = 16
    bias = nn.Parameter(torch.Tensor(out_channels))                

    offset = torch.randn(1, 288, 180, 320)
    mask = torch.randn(1, 144, 180, 320)

    deform_conv2d(a, offset, weight, bias, stride, padding, dilation, mask)


if __name__ == '__main__':
    main()