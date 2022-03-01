from typing import Optional, Any, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from common import benchmark


class StochasticDepth(nn.Module):
    """Randomly drop a layer"""
    def __init__(self, module: nn.Module, survival_rate: float = 1.) -> None:
        super().__init__()
        self.module = module
        self._drop = torch.distributions.Bernoulli(torch.tensor(1 - survival_rate))
    
    def forward(self, x: Tensor) -> Tensor:
        return 0 if self.training and self._drop.sample() else self.module(x)


class LayerNorm(nn.LayerNorm):
    """Permute the input tensor so that the channel dimension is the last one."""
    def __init__(self, num_features: int, eps: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(num_features, eps=eps, **kwargs)
    
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def dw_conv7x7(in_planes: int, out_planes: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 7, padding=3, groups=in_planes, **kwargs)

def conv1x1(in_planes: int, out_planes: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, **kwargs)

def patch_conv(in_planes: int, out_planes: int, patch_size: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, patch_size, stride=patch_size, **kwargs)


class CNBlock(nn.Module):
    expansion: int = 4

    def __init__(self, planes: int, stodepth_survive: float = 1.0) -> None:
        super().__init__()

        expand_width = planes * self.expansion
        main = nn.Sequential(
            dw_conv7x7(planes, planes),
            LayerNorm(planes),
            conv1x1(planes, expand_width),
            nn.GELU(),
            conv1x1(expand_width, planes)
        )
        self.main = main if stodepth_survive == 1.0 else StochasticDepth(main, stodepth_survive)
        
    def forward(self, x: Tensor) -> Tensor:
        return x + self.main(x)


class ConvNext(nn.Module):
    def __init__(
        self,
        layers: List[int],
        widths: List[int],
        num_classes: int = 1000,
        stodepth_survive: Optional[float] = 1.
    ) -> None:
        assert len(layers) == len(widths) == 4, "Length of layers and widths param must be 4"
        super().__init__()

        
        # Patchify downsampling stem
        stem = nn.Sequential(
            patch_conv(3, widths[0], patch_size=4),
            LayerNorm(widths[0], eps=1e-6)
        )

        # Res1 -> Res4
        stages = []
        for layer, width in zip(layers, widths):
            stages.append(
                nn.Sequential(
                    *[CNBlock(width, stodepth_survive) for _ in range(layer)]
                )
            )
        self.stages = nn.ModuleList(stages)

        # Intermediate downsampling layers
        ds_layers = [stem]
        for cur_width, next_width in zip(widths, widths[1:]):
            ds_layers.append(
                nn.Sequential(
                    LayerNorm(cur_width, eps=1e-6),
                    patch_conv(cur_width, next_width, patch_size=2)
                )
            )
        self.ds_layers = nn.ModuleList(ds_layers)

        # Pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.n2 = LayerNorm(widths[-1], eps=1e-6)
        self.fc = nn.Linear(widths[-1], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)


    def _forward_impl(self, x: Tensor) -> Tensor:
        for ds_layer, stage in zip(self.ds_layers, self.stages):
            x = ds_layer(x)
            x = stage(x)

        x = self.avgpool(x)
        x = self.n2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(layers: List[int], widths: List[int], **kwargs: Any) -> ConvNext:
    model = ConvNext(layers, widths, **kwargs)
    return model


def convnext_t(**kwargs: Any) -> ConvNext:
    return _convnext([3, 3, 9, 3], [96, 192, 384, 768], **kwargs)

def convnext_s(**kwargs: Any) -> ConvNext:
    return _convnext([3, 3, 27, 3], [96, 192, 384, 768], **kwargs)

def convnext_b(**kwargs: Any) -> ConvNext:
    return _convnext([3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)

def convnext_l(**kwargs: Any) -> ConvNext:
    return _convnext([3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)

def convnext_xl(**kwargs: Any) -> ConvNext:
    return _convnext([3, 3, 27, 3], [256, 512, 1024, 2048], **kwargs)


if __name__ == "__main__":
    model = convnext_t(stodepth_survive=1.0)
    benchmark(model)
    print(model)