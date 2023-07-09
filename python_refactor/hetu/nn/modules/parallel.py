import hetu
from .module import Module
import numpy as np

from typing import Any

__all__ = [
    'ColumnParallelLinear', 
    'RowParallelLinear', 
]

np.random.seed(2023)

def xavier_normal_(size, gain=1.0):
    shape = size
    if len(shape) < 2:
        raise ValueError("Shape must have at least 2 dimensions")

    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    fan_out = shape[1] if len(shape) == 2 else shape[0]

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(0, std, size=shape)

# assume that device_group across different pp stages has the same device num,
# so we can use device_index in current device_group to generate the dummy data 
# for other device_group, which will not be used in exec runtime 
def parallel_data_provider(ds, device_index, init_func='np.random.normal', shape=None, data=None):
    if init_func == 'data':
        assert data is not None, "data must be provided in mode {init_func}!"
        global_data = data
    else:
        assert shape is not None, "shape must be provided for generating data in mode {init_func}!"
        global_data = eval(init_func)(size=shape) # default data generate: from np.random.normal

    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        local_data = local_data.take(range(start, stop), axis=dim)
    return local_data

# process: x->dup, w->split1 => y->split1 => y->dup
class ColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, device_group, local_device_index=None, 
                 bias=True, gather_output=True, init_method='xavier_normal_'):
        super(ColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_group = device_group
        self.gather_output = gather_output

        num_devices = device_group.num_devices
        local_device = hetu.local_device()
        if device_group.contains(local_device):
            device_index = device_group.get_index(local_device)
        else: # for pipeline parallel
            assert local_device_index is not None, "local_device_index should be assigned when device_group doesn't contain local_device!"
            device_index = local_device_index 
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        ds_split0 = hetu.DistributedStates(num_devices, {0: num_devices}, [0])
        self.ds_map = {'dup': ds_dup, 'split0': ds_split0}
        # dup [4,8] -> [2,8] + [2,8]
        # local init: if dup, extra comm
        self.weight = hetu.Tensor(parallel_data_provider(ds_split0, device_index, init_func=init_method, shape=(out_features, in_features)),
                                  dtype=hetu.float32, requires_grad=True, ds=ds_split0, device_group=device_group, name='w_colparallel')
        if bias:
            self.bias = hetu.Tensor(parallel_data_provider(ds_split0, device_index, init_func='data', data=np.zeros(out_features)),
                                   dtype=hetu.float32, requires_grad=True, ds=ds_split0, device_group=device_group, name='bias_colparallel')
        else:
            self.bias = None
      
    def forward(self, input):
        ds_input = input.distributed_states
        if ds_input.check_equal(self.ds_map['dup']):
            input_dup = input
        else:
            input_dup = hetu.comm(input, self.ds_map['dup'], name='comm_for_gather_input_colparallel')
        output_split1 = hetu.linear(input_dup, self.weight, self.bias, trans_b=True, name='linear_for_colparallel')
        if not self.gather_output:
            output = output_split1
            bias = self.bias
        else:
            output = hetu.comm(output_split1, self.ds_map['dup'], name='comm_for_gather_output_colparallel')
            bias = hetu.comm(self.bias, self.ds_map['dup'], name='comm_for_gather_bias_colparallel')
        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class RowParallelLinear(Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    """
    def __init__(self, in_features, out_features, device_group, 
                 local_device_index=None, bias=True, init_method='xavier_normal_'):
        super(RowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device_group = device_group

        num_devices = device_group.num_devices
        local_device = hetu.local_device()
        if device_group.contains(local_device):
            device_index = device_group.get_index(local_device)
        else: # for pipeline parallel
            assert local_device_index is not None, "local_device_index should be assigned when device_group doesn't contain local_device!"
            device_index = local_device_index 
        ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1])
        ds_split1 = hetu.DistributedStates(num_devices, {1: num_devices}, [1])
        self.ds_map = {'dup': ds_dup, 'split1': ds_split1}
        self.weight = hetu.Tensor(parallel_data_provider(ds_split1, device_index, init_func=init_method, shape=(out_features, in_features)),
                                  dtype=hetu.float32, requires_grad=True, ds=ds_split1, device_group=device_group, name='w_rowparallel')
        if bias:
            self.bias = hetu.Tensor(parallel_data_provider(ds_dup, device_index, init_func='data', data=np.zeros(out_features)),
                                   dtype=hetu.float32, requires_grad=True, ds=ds_dup, device_group=device_group, name='bias_rowparallel')
        else:
            self.bias = None

    def forward(self, input):
        ds_input = input.distributed_states
        if ds_input.check_equal(self.ds_map['split1']):
            input_split1 = input
        else:
            input_split1 = hetu.comm(input, self.ds_map['split1'], name='comm_for_split_input_rowparallel')

        output_partial = hetu.linear(input_split1, self.weight, trans_b=True, name='linear_for_rowparallel')
        output_dup = hetu.comm(output_partial, self.ds_map['dup'], name='comm_for_reduce_output_rowparallel')
        # make allreduce for x*w^T first, then add bias, to ensure that bias will 
        # be updated the same among devices, correspond to ds_dup for bias
        output = output_dup + self.bias if self.bias is not None else output_dup
        return output