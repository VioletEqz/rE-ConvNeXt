import torch

from typing import Any, Callable


def get_model_factory(repository: Any,
                      model_name: str) -> Callable[..., torch.nn.Module]:
    if hasattr(repository, model_name):
        return getattr(repository, model_name)