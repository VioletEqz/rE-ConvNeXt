from typing import Optional, Type, Any, List

import torch
import torch.nn as nn
from torch import Tensor

from .common import StochasticModule, benchmark, conv3x3, conv1x1


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1
    ) -> None:
        super().__init__()
        
        norm_layer = nn.BatchNorm2d
        self.act = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(inplanes, planes)
        self.n1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.n2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.n3 = norm_layer(planes * self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.n1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.n2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.n3(out)

        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        main_path: nn.Module,
        projection: Optional[nn.Module] = None,
        stodepth_survival_rate: float = 1.
    ) -> None:
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        self.main_path = StochasticModule(main_path, stodepth_survival_rate) \
                         if stodepth_survival_rate < 1. else main_path
        
        self.projection = projection

    def forward(self, x: Tensor) -> Tensor:
        out = self.main_path(x)
        identity = self.projection(x) if self.projection is not None else x
        out += identity
        out = self.act(out)
        
        return out


class ConvNext(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        stodepth_survival_rate: float = 1.
    ) -> None:
        super().__init__()
        
        self.stodepth_survival_rate = stodepth_survival_rate
        self.norm_layer = nn.BatchNorm2d

        # NOTE: Patchify downsampling stem
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=4, stride=4, padding=0, bias=False)
        self.n1 = nn.BatchNorm2d(self.inplanes)

        # Res1 -> Res4
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Pooling and FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: Type[Bottleneck], planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self.norm_layer

        layers = []
        
        # First block of the layer
        if stride != 1 or self.inplanes != planes * block.expansion:
            projection = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )
        else:
            projection = None
        layers.append(
            ResBlock(
                block(self.inplanes, planes, stride=stride),
                projection=projection,
                stodepth_survival_rate=self.stodepth_survival_rate
            )
        )

        # Remaining blocks of the layer
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                ResBlock(
                    block(self.inplanes, planes, stride=1),
                    projection=None,
                    stodepth_survival_rate=self.stodepth_survival_rate
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.n1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _convnext(block: Type[Bottleneck], layers: List[int], **kwargs: Any) -> ConvNext:
    model = ConvNext(block, layers, **kwargs)
    return model


# NOTE: New stage ratios
def convnext_t(**kwargs: Any) -> ConvNext:
    return _convnext(Bottleneck, [3, 3, 9, 3], **kwargs)

def convnext_s(**kwargs: Any) -> ConvNext:
    return _convnext(Bottleneck, [3, 3, 27, 3], **kwargs)


if __name__ == "__main__":
    model = convnext_t(stodepth_survival_rate=0.9)
    benchmark(model)