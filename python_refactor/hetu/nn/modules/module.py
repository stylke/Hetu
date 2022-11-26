import hetu
from hetu import Tensor
from ..parameter import Parameter
from collections import OrderedDict, namedtuple

from typing import Union, Tuple, List, Dict, Set, Any, Callable, Iterator, Optional, TypeVar
from hetu import _tensor_type_t
from ..parameter import _param_type_t

_module_type_t = TypeVar('T', bound='Module')
_member_type_t = Union[_param_type_t, _module_type_t, _tensor_type_t]
_member_t = Union[Optional[Parameter], Optional['Module'], Optional[Tensor]]

class Module(object):

    def __init__(self):
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_buffers", OrderedDict())
        super().__setattr__("_modules", OrderedDict())
        super().__setattr__("_non_persistent_buffers_set", set())
    
    def __getattr__(self, name: str) -> Any:
        _parameters = self.__dict__.get('_parameters')
        if _parameters is not None:
            param = _parameters.get(name)
            if param is not None:
                return param
        _buffers = self.__dict__.get('_buffers')
        if _buffers is not None:
            buffers = _buffers.get(name)
            if buffers is not None:
                return buffers
        _modules = self.__dict__.get('_modules')
        if _modules is not None:
            module = _modules.get(name)
            if module is not None:
                return module
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Any) -> None:
        def remove_from_members(*members):
            for dict_or_set in members:
                if dict_or_set is not None and name in dict_or_set:
                    if isinstance(dict_or_set, dict):
                        del dict_or_set[name]
                    else:
                        dict_or_set.discard(name)
        
        _parameters = self.__dict__.get('_parameters')
        _modules = self.__dict__.get('_modules')
        _buffers = self.__dict__.get('_buffers')
        _non_persistent_buffers_set = self.__dict__.get('_non_persistent_buffers_set')
        
        # Parameters
        if isinstance(value, Parameter):
            remove_from_members(self.__dict__, _modules, _buffers, 
                                _non_persistent_buffers_set)
            self.register_parameter(name, value)
            return
        
        # Modules
        if isinstance(value, Module):
            remove_from_members(self.__dict__, _parameters, _buffers, 
                                _non_persistent_buffers_set)
            self.register_module(name, value)
            return
        
        # Buffers
        if isinstance(value, Tensor):
            remove_from_members(self.__dict__, _parameters, _modules)
            self.register_buffer(name, value, persistent=True)
            return
        
        remove_from_members(self.__dict__, _parameters, _modules, _buffers, 
                            _non_persistent_buffers_set)
        super().__setattr__(name, value)
    
    def __delattr__(self, name: str) -> None:
        _parameters = self.__dict__.get('_parameters')
        if _parameters is not None and name in _parameters:
            del _parameters[name]
            return
        
        _modules = self.__dict__.get('_modules')
        if _modules is not None and name in _modules:
            del _modules[name]
            return
        
        _buffers = self.__dict__.get('_buffers')
        if _buffers is not None and name in _buffers:
            del _buffers[name]
            _non_persistent_buffers_set = self.__dict__.get('_non_persistent_buffers_set')
            del _non_persistent_buffers_set[name]
            return
        
        super().__delattr__(name)
        
    
    def __dir__(self):
        module_attrs = dir(self.__class__)
        self_attrs = list(self.__dict__.keys())
        _parameters = list(self._parameters.keys())
        _modules = list(self._modules.keys())
        _buffers = list(self._buffers.keys())
        keys = module_attrs + self_attrs + _parameters + _modules + _buffers
        return sorted(keys)
    
    ############################################################################
    # Registration of members
    ############################################################################
    
    def _register_member(self, name: str, value: _member_t, members: dict, 
                         reg_type: _member_type_t) -> None:
        if not isinstance(name, str):
            raise TypeError(
                f"Name of {reg_type.__name__} must be string "
                f"(got {type(name).__name__})")
        if name == '':
            raise KeyError(f"Name of {reg_type.__name__} must not be empty")
        if '.' in name:
            raise KeyError(
                f"Name of {reg_type.__name__} must not contain \".\" "
                f"(got \"{name}\")")
        
        if value is None:
            members[name] = None
        elif not isinstance(value, reg_type):
            raise TypeError(
                f"Cannot register a '{type(value).__name__}' object as "
                f"a {reg_type.__name__} object")
        else:
            members[name] = value
    
    def register_parameter(self, name: str, value: Optional[Parameter]) -> None:
        _parameters = self.__dict__.get('_parameters')
        if _parameters is None:
            raise AttributeError(
                "Cannot register parameters before calling Module.__init__()")
        self._register_member(name, value, _parameters, Parameter)
    
    def register_module(self, name: str, value: Optional['Module']) -> None:
        _modules = self.__dict__.get('_modules')
        if _modules is None:
            raise AttributeError(
                "Cannot register modules before calling Module.__init__()")
        self._register_member(name, value, _modules, Module)
    
    def add_module(self, name: str, value: Optional['Module']) -> None:
        self.register_module(name, value)
    
    def register_buffer(self, name: str, value: Optional[Tensor], 
                        persistent: bool = True) -> None:
        _buffers = self.__dict__.get('_buffers')
        if _buffers is None:
            raise AttributeError(
                "Cannot register buffers before calling Module.__init__()")
        self._register_member(name, value, _buffers, Tensor)
        if persistent:
            self._non_persistent_buffers_set.discard(name)
        else:
            self._non_persistent_buffers_set.add(name)

    ############################################################################
    # Iterator/Generator of members
    ############################################################################
    
    def _named_members(self, get_members_fn: Callable, prefix: str = '', 
                       recurse: bool = True) -> Iterator[Tuple[str, Any]]:
        visited = set()
        if recurse:
            modules = self.named_modules(prefix=prefix)
        else:
            modules = [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in visited:
                    continue
                visited.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        return self._named_members(
            lambda m: m._parameters.items(), 
            prefix=prefix, 
            recurse=recurse)
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param
    
    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', 
                      remove_duplicate: bool = True) -> Iterator[Tuple[str, 'Module']]:
        memo = memo or set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is not None:
                    sub_prefix = prefix + ('.' if prefix else '') + name
                    for m in module.named_modules(memo, sub_prefix, remove_duplicate):
                        yield m
    
    def modules(self) -> Iterator['Module']:
        for _, module in self.named_modules():
            yield module
    
    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        return self._named_members(
            lambda m: m._buffers.items(), 
            prefix=prefix, 
            recurse=recurse)
    
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for _, buffer in self.named_buffers(recurse=recurse):
            yield buffer
    
    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        memo = set()
        for name, child in self._modules.items():
            if child is not None and child not in memo:
                memo.add(child)
                yield name, child

    def children(self) -> Iterator['Module']:
        for _, child in named_children:
            yield child
    
    ############################################################################
    # Call and Forward
    ############################################################################

    def __call__(self, *input, **kwargs) -> Any:
        return self.forward(*input, **kwargs)
    
    def forward(self, *input: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"Forward of module '{type(self).__name__}' is not implemented")
