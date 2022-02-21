import torch
import torch.nn as nn
from torch import Tensor


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, depthwise: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    groups = in_planes if depthwise else 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class StochasticModule(nn.Module):
    """
    Stochastic Depth: https://arxiv.org/abs/1603.09382
    Randomly drop layers in each forward pass for the whole input batch
    """
    def __init__(
        self,
        module: nn.Module,
        survival_rate: float = 1.
    ) -> None:
        super().__init__()
        self.module = module
        self._drop = torch.distributions.Bernoulli(torch.tensor(1 - survival_rate))
    
    def forward(self, x: Tensor) -> Tensor:
        return 0 if self.training and self._drop.sample() else self.module(x)