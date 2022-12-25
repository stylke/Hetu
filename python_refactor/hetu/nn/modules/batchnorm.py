import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'NormBase',
    'BatchNorm',
]

class NormBase(Module):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super(NormBase, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = hetu.nn.Parameter(hetu.ones([num_features]))
        self.bias = hetu.nn.Parameter(hetu.zeros([num_features]))

class BatchNorm(NormBase):

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super(BatchNorm, self).__init__(num_features, eps, momentum)

    def forward(self, input: Tensor) -> Tensor:
        return hetu.batch_norm(input, self.weight, self.bias, self.momentum, self.eps)

