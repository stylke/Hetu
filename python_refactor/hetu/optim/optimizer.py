import hetu
from hetu import Tensor
from typing import Optional, Iterable

class Optimizer(object):

    def __init__(self, params: Optional[Iterable[Tensor]], defaults: dict):
        self.params = params
        if self.params is not None:
            self.params = list(self.params)
            assert len(self.params) > 0, "No variables are provided"
            for p in self.params:
                assert p.trainable, f"Parameter {p} is not trainable"
        
        self.defaults = defaults
    
    def zero_grad(self):
        for p in self.params:
            p.zero_grad()
    
    def step(self):
        update_ops = []
        for p in self.params:
            if p.grad is None:
                continue
            update_op = self.apply_dense(p, p.grad)
            update_ops.append(update_op)
        hetu.group(update_ops).get_or_compute()
    
    def apply_dense(self, param, grad):
        raise NotImplementedError
