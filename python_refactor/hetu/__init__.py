# TODO: output the library to `hetu` directory
import _hetu_core
from _hetu_core import *

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
_tensor_type_t = TypeVar('T', bound='Tensor')

from hetu import nn as nn
from hetu import optim as optim

import builtins # bool is resovled as hetu.bool

class _Context(object):
    def __init__(self, devices=None, stream_index=None, extra_deps=None):
        self.devices = devices
        self.stream_index = stream_index
        self.extra_deps = extra_deps
        if devices is not None:
            self.devices = DeviceGroup(devices)
        if stream_index is not None:
            if ((not isinstance(stream_index, builtins.int)) or 
                isinstance(stream_index, builtins.bool)):
                raise ValueError(
                    "Expected 'int' type for stream index, got "
                    f"'{type(stream_index).__name__}'")
        if extra_deps is not None:
            if not isinstance(extra_deps, (list, tuple)):
                raise ValueError("Extra dependencies should be lists or tuples of tensors")
            for x in extra_deps:
                if not isinstance(x, Tensor):
                    raise ValueError(f"'{type(x).__name__}' object is not a tensor")
            self.extra_deps = [group(extra_deps)]

    def __enter__(self):
        _hetu_core._internal_contxt.push_op_ctx(
            device_group=self.devices, 
            stream_index=self.stream_index, 
            extra_deps=self.extra_deps)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_contxt.pop_op_ctx(
            pop_device_group=(self.devices is not None), 
            pop_stream_index=(self.stream_index is not None), 
            pop_extra_deps=(self.extra_deps is not None))

def context(devices=None, stream_index=None, extra_deps=None):
    return _Context(devices=devices, stream_index=stream_index, extra_deps=extra_deps)

def control_dependencies(control_inputs):
    return _Context(extra_deps=control_inputs)
