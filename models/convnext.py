import math
from typing import Optional, Any, List

import torch
import torch.nn as nn
from torch import Tensor


class StochasticDepth(nn.Module):
    """Randomly drop a module"""
    def __init__(self, module: nn.Module, survival_rate: float = 1.) -> None:
        super().__init__()
        self.module = module
        self.survival_rate = survival_rate
        self._drop = torch.distributions.Bernoulli(torch.tensor(1 - survival_rate))
    
    def forward(self, x: Tensor) -> Tensor:
        return 0 if self.training and self._drop.sample() else self.module(x)
    
    def __repr__(self) -> str:
        return self.module.__repr__() + f", stodepth_survival_rate={self.survival_rate:.2f}"


class LayerNorm(nn.LayerNorm):
    """Permute the input tensor so that the channel dimension is the last one."""
    def __init__(self, num_features: int, eps: float = 1e-6, **kwargs: Any) -> None:
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
        self.stem = nn.Sequential(
            patch_conv(3, widths[0], patch_size=4),
            LayerNorm(widths[0])
        )

        # Stage 1 -> 4 and intermediate downsampling layers
        for idx, (layer, width) in enumerate(zip(layers, widths)):
            self.add_module(
                f"stage{idx + 1}",
                nn.Sequential(*[CNBlock(width, stodepth_survive) for _ in range(layer)])
            )
            if idx == 3: break
            self.add_module(
                f"ds{idx + 1}",
                nn.Sequential(
                    LayerNorm(width),
                    patch_conv(width, widths[idx + 1], patch_size=2)
                )
            )

        # Pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = LayerNorm(widths[-1])
        self.fc = nn.Linear(widths[-1], num_classes
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.stage1(x)
        x = self.ds1(x)
        x = self.stage2(x)
        x = self.ds2(x)
        x = self.stage3(x)
        x = self.ds3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = self.norm(x)
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
    from .common import benchmark
    model = convnext_t()
    benchmark(model)