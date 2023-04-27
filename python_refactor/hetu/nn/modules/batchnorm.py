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
        # self.weight = hetu.nn.Parameter(hetu.ones([num_features], trainable=True))
        # self.bias = hetu.nn.Parameter(hetu.zeros([num_features], trainable=True))

class BatchNorm(NormBase):
    #TODO:Normalize operators should have only one output.Now we have three. 

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super(BatchNorm, self).__init__(num_features, eps, momentum)
        self.weight = hetu.nn.Parameter(hetu.ones([num_features], trainable=True))
        self.bias = hetu.nn.Parameter(hetu.zeros([num_features], trainable=True))
        self.running_mean = hetu.nn.Parameter(hetu.empty([num_features], trainable=False))
        self.running_var = hetu.nn.Parameter(hetu.empty([num_features], trainable=False))
        # self.save_mean = hetu.nn.Parameter(hetu.empty([num_features], trainable=False))
        # self.save_var = hetu.nn.Parameter(hetu.empty([num_features], trainable=False))

    def forward(self, input: Tensor) -> Tensor:
        # tmp_weight = hetu.nn.Parameter(hetu.ones([self.num_features], trainable=True))
        # tmp_bias = hetu.nn.Parameter(hetu.zeros([self.num_features], trainable=True))
        return hetu.batch_norm(input, self.weight, self.bias, self.running_mean, self.running_var, self.momentum, self.eps)[0]

