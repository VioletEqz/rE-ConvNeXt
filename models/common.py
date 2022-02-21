import torch.nn as nn


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, depthwise: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    groups = in_planes if depthwise else 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)