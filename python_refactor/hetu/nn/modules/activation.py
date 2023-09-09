import hetu
from hetu import Tensor
from .module import Module
import math
from .utils import _pair

from typing import Any, TypeVar, Union, Tuple, Optional

__all__ = [
    'ReLU', 
    'Sigmoid', 
    'Tanh', 
    'LeakyReLU',
]

class ReLU(Module):

    def __init__(self, inplace: bool = False):
        with hetu.graph("define_and_run"):
            super(ReLU, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return hetu.relu(input)


class Sigmoid(Module):

    def __init__(self, inplace: bool = False):
        with hetu.graph("define_and_run"):
            super(Sigmoid, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hetu.graph("define_and_run"):
            return hetu.sigmoid(input)

class Tanh(Module):

    def __init__(self, inplace: bool = False):
        with hetu.graph("define_and_run"):
            super(Tanh, self).__init__()
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hetu.graph("define_and_run"):
            return hetu.tanh(input)


class LeakyReLU(Module):

    def __init__(self, negative_slope: float = 0.1, inplace: bool = False):
        with hetu.graph("define_and_run"):
            super(LeakyReLU, self).__init__()
            self.negative_slope = negative_slope
            self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        with hetu.graph("define_and_run"):
            return hetu.leakyrelu(input, self.negative_slope)