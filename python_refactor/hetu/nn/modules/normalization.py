import hetu
from hetu import Tensor
import numbers
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional, List

__all__ = [
    'LayerNorm',
]

class LayerNorm(Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.weight = hetu.nn.Parameter(hetu.ones(self.normalized_shape))
        self.bias = hetu.nn.Parameter(hetu.zeros(self.normalized_shape))

    def forward(self, input: Tensor) -> Tensor:
        return hetu.layernorm(input, self.weight, self.bias, self.eps)

