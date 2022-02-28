import torch
import torch.nn as nn
from torch import Tensor
from ptflops import get_model_complexity_info

def benchmark(model, input_size=(3, 224, 224), print_layers=False):
    macs, params = get_model_complexity_info(model, input_size, as_strings=True,
                                             print_per_layer_stat=print_layers, verbose=False)
    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, depthwise: bool = False, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    groups = in_planes if depthwise else 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=bias)

def conv7x7(in_planes: int, out_planes: int, stride: int = 1, depthwise: bool = False, bias: bool = False) -> nn.Conv2d:
    """3x3 convolution with padding"""
    groups = in_planes if depthwise else 1
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, groups=groups, bias=bias)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def downsample(in_planes: int, out_planes: int, kernel_size=2, stride=2, eps=1e-6) -> nn.Sequential:
    return nn.Sequential(
                        nn.LayerNorm(in_planes, eps=eps),
                        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride),)

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