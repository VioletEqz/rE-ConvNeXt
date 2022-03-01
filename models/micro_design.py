from typing import Optional, Type, Any, List

import torch
import torch.nn as nn
from torch import Tensor

from .common import StochasticModule, benchmark, conv1x1, conv7x7, downsample, LayerNorm


class InvertedBottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, planes: int) -> None:
        super().__init__()
        
        # NOTE: Replacing BatchNorm2d with LayerNorm
        norm_layer = LayerNorm
        
        # NOTE:Replacing RELU with GELU
        self.act = nn.GELU()

        expand_width = planes * self.expansion
        # NOTE: Removed an activation function on the first convolutional layer
        # NOTE: Removed two normalization layers after the first convolutional layer
        self.conv1 = conv7x7(planes, planes, depthwise=True)
        self.n1 = norm_layer(planes)
        self.conv2 = conv1x1(planes, expand_width, bias=True)
        self.conv3 = conv1x1(expand_width, planes, bias=True)
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.n1(out)

        out = self.conv2(out)
        out = self.act(out)

        out = self.conv3(out)

        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        main_path: nn.Module,
        projection: Optional[nn.Module] = None,
        stodepth_survival_rate: float = 1.
    ) -> None:
        super().__init__()

        # NOTE: Removed the activation function on the out path of the block
        self.main_path = StochasticModule(main_path, stodepth_survival_rate) \
                         if stodepth_survival_rate < 1. else main_path
        
        self.projection = projection

    def forward(self, x: Tensor) -> Tensor:
        out = self.main_path(x)
        identity = self.projection(x) if self.projection is not None else x
        out += identity
        
        return out


class ConvNext(nn.Module):
    def __init__(
        self,
        block: InvertedBottleneck,
        layers: List[int],
        width: List[int],
        num_classes: int = 1000,
        stodepth_survival_rate: float = 1.
    ) -> None:
        super().__init__()
        
        self.stodepth_survival_rate = stodepth_survival_rate
        # NOTE: Changed the norm layer to LayerNorm
        norm_layer = LayerNorm

        # Patchify downsampling stem
        self.inplanes = width[0]
        self.conv1 = nn.Conv2d(3, width[0], kernel_size=4, stride=4, padding=0, bias=False)
        # NOTE: Changed the norm layer to LayerNorm
        self.n1 = norm_layer(width[0])

        # Res1 -> Res4 with custom widths
        # NOTE: Added a downsample layer before every stage except the first one, which
        # uses the stem instead.
        self.layer1 = self._make_layer(block, width[0], layers[0])
        self.downsample1 = downsample(width[0], width[1])
        self.layer2 = self._make_layer(block, width[1], layers[1])
        self.downsample2 = downsample(width[1], width[2])
        self.layer3 = self._make_layer(block, width[2], layers[2])
        self.downsample3 = downsample(width[2], width[3])
        self.layer4 = self._make_layer(block, width[3], layers[3])

        # Pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # NOTE: added a LayerNorm after global average pooling
        self.n2 = norm_layer(width[3])
        self.fc = nn.Linear(width[3], num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[InvertedBottleneck], planes: int, num_blocks: int) -> nn.Sequential:
        layers = []
        for _ in range(num_blocks):
            layers.append(
                ResBlock(
                    block(planes),
                    projection=None,
                    stodepth_survival_rate=self.stodepth_survival_rate
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.n1(x)

        # NOTE: Adding the new layers to the forward function pipeline
        x = self.layer1(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self.layer3(x)
        x = self.downsample3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.n2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(block: Type[InvertedBottleneck], layers: List[int], width: List[int], **kwargs: Any) -> ConvNext:
    model = ConvNext(block, layers, width, **kwargs)
    return model


def convnext_t(**kwargs: Any) -> ConvNext:
    return _convnext(InvertedBottleneck, [3, 3, 9, 3], [96, 192, 384, 768], **kwargs)

def convnext_s(**kwargs: Any) -> ConvNext:
    return _convnext(InvertedBottleneck, [3, 3, 27, 3], [96, 192, 384, 768], **kwargs)

def convnext_b(**kwargs: Any) -> ConvNext:
    return _convnext(InvertedBottleneck, [3, 3, 27, 3], [128, 256, 512, 1024], **kwargs)

def convnext_l(**kwargs: Any) -> ConvNext:
    return _convnext(InvertedBottleneck, [3, 3, 27, 3], [192, 384, 768, 1536], **kwargs)

def convnext_xl(**kwargs: Any) -> ConvNext:
    return _convnext(InvertedBottleneck, [3, 3, 27, 3], [256, 512, 1024, 2048], **kwargs)


if __name__ == "__main__":
    model = convnext_t(stodepth_survival_rate=1.0)
    benchmark(model)
    # print(model)