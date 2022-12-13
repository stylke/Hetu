import hetu
from hetu import Tensor

from typing import TypeVar
_param_type_t = TypeVar('T', bound='Parameter')

class Parameter(Tensor):
    
    def __new__(cls, data: Tensor, trainable: bool = True):
        # TODO: support None?
        assert data is not None, "Cannot create Parameter from None"
        if type(data) is Tensor or type(data) is Parameter:
            return Tensor._make_subclass(cls, data.to_variable(trainable), trainable)
        else:
            raise TypeError(
                f"Cannot create Parameter using data of "
                f"type {type(data).__name__}")

    def __repr__(self) -> str:
        return "Parameter containing:\n" + super(Parameter, self).__repr__()

class UninitializedTensorMixin:

    def materialize(self, shape, device=None, dtype=None):
        if device is None:
            device = self.data.device
        if dtype is None:
            dtype = self.data.dtype
        self.data = hetu.empty(shape, device=device, dtype=dtype)
        self.__class__ = self.cls_to_become

    @property
    def shape(self):
        raise RuntimeError(
            'Can\'t access the shape of an uninitialized parameter or buffer. '
            'This error usually happens in `load_state_dict` when trying to load '
            'an uninitialized parameter into an initialized one. '
            'Call `forward` to initialize the parameters before accessing their attributes.')

    def share_memory_(self):
        raise RuntimeError(
            'Can\'t share memory on an uninitialized parameter or buffer. '
            'Call `forward` to initialize the parameters before calling '
            '`module.share_memory()`.')

    def __repr__(self):
        return f'<{self.__class__.__name__}>'


def is_lazy(param):
    return isinstance(param, UninitializedTensorMixin)

class UninitializedParameter(UninitializedTensorMixin, Parameter):

    cls_to_become = Parameter


