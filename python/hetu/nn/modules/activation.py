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
    'NewGeLU'
]


class ReLU(Module):

    def __init__(self, multi_ds_parallel_config = None, inplace: bool = False):
        super(ReLU, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.inplace:
            return hetu.relu_(input)
        else:
            return hetu.relu(input)


class Sigmoid(Module):

    def __init__(self, multi_ds_parallel_config = None, inplace: bool = False):
        super(Sigmoid, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.inplace:
            return hetu.sigmoid_(input)
        else:
            return hetu.sigmoid(input)


class Tanh(Module):

    def __init__(self, multi_ds_parallel_config = None, inplace: bool = False):
        super(Tanh, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.inplace:
            return hetu.tanh_(input)
        else:
            return hetu.tanh(input)


class LeakyReLU(Module):

    def __init__(self, negative_slope: float = 0.1, multi_ds_parallel_config = None, inplace: bool = False):
        super(LeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.ds_parallel_configs = multi_ds_parallel_config
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if self.inplace:
            return hetu.leakyrelu_(input, self.negative_slope)
        else:
            return hetu.leakyrelu(input, self.negative_slope)
      
        
class NewGeLU(Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    def __init__(self, multi_ds_parallel_config = None, inplace: bool = False):
        super(NewGeLU, self).__init__()
        self.ds_parallel_configs = multi_ds_parallel_config
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        #  0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        # TODO: implement hetu.pow(input, 3.0) to replace input * input * input, or implement a cuda kernel
        return 0.5 * input * (1.0 + hetu.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input * input * input))))