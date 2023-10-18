# TODO: output the library to `hetu` directory
import _hetu_core
from _hetu_core import *

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
_tensor_type_t = TypeVar('T', bound='Tensor')

from hetu import nn as nn
from hetu import utils as utils
# from hetu import optim as optim

import builtins # bool is resovled as hetu.bool

cur_graph_contexts = []

class _OpContext(object):
    def __init__(self, eager_device=None, devices=None, stream_index=None, extra_deps=None):
        self.eager_device = eager_device
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
        _hetu_core._internal_context.push_op_ctx(
            eager_device=self.eager_device,
            device_group=self.devices, 
            stream_index=self.stream_index, 
            extra_deps=self.extra_deps)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_op_ctx(
            pop_eager_device=(self.eager_device is not None),
            pop_device_group=(self.devices is not None), 
            pop_stream_index=(self.stream_index is not None), 
            pop_extra_deps=(self.extra_deps is not None))

def context(eager_device=None, devices=None, stream_index=None, extra_deps=None):
    return _OpContext(eager_device=eager_device, devices=devices, stream_index=stream_index, extra_deps=extra_deps)

def control_dependencies(control_inputs):
    return _OpContext(extra_deps=control_inputs)

class _GraphContext(object):
    def __init__(self, g):
        if isinstance(g, Graph):
            self.graph = g
        elif isinstance(g, builtins.str):
            if g == "eager":
                self.graph = _hetu_core._internal_context.get_default_eager_graph()
            elif g == "define_and_run":
                self.graph = _hetu_core._internal_context.get_default_define_and_run_graph()
            elif g == "define_by_run":
                self.graph = _hetu_core._internal_context.get_default_define_by_run_graph()
            else:
                self.graph = _hetu_core._internal_context.get_graph(g)
        elif isinstance(g, builtins.int):
            self.graph = _hetu_core._internal_context.get_graph(g)
        else:
            raise ValueError(f"Cannot parse type '{type(g).__name__}' as hetu.Graph")

    def __enter__(self):
        if len(cur_graph_contexts) > 0:
            self.wrapped_context = True
            return cur_graph_contexts[0]
        self.wrapped_context = False
        _hetu_core._internal_context.push_graph_ctx(self.graph.id)
        cur_graph_contexts.append(self)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        if self.wrapped_context:
            return
        _hetu_core._internal_context.pop_graph_ctx()
        cur_graph_contexts.remove(self)

def graph(g):
    return _GraphContext(g)

class _AutocastContext(object):
    def __init__(self):
        self.autocast = _hetu_core._internal_context.get_default_autocast()

    def __enter__(self):
        _hetu_core._internal_context.push_autocast_ctx(self.autocast.id)
        return self
    
    def __exit__(self, e_type, e_value, e_trace):
        _hetu_core._internal_context.pop_autocast_ctx()

def autocast():
    return _AutocastContext()
