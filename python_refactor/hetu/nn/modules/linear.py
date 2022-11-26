import hetu
from hetu import Tensor
from .module import Module
import math

from typing import Any

__all__ = [
    'Identity', 
    'Linear', 
]

class Identity(Module):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return input

class Linear(Module):
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = hetu.nn.Parameter(hetu.empty([out_features, in_features]))
        if bias:
            self.bias = hetu.nn.Parameter(hetu.empty([out_features]))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        hetu.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = hetu.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1. / math.sqrt(fan_in) if fan_in > 0 else 0
            hetu.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input: Tensor) -> Tensor:
        return hetu.linear(input, self.weight, self.bias)
