import hetu
from .module import Module
import numbers

__all__ = [
    'HtMultiColumnParallelLinear', 
    'HtMultiRowParallelLinear', 
    'HtMultiParallelEmbedding',
    'HtMultiVocabParallelEmbedding',
    'HtMultiParallelLayerNorm',
]

def parallel_data_provider(global_data, ds, device_index):
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
  
def parallel_multi_data_provider(global_data, multi_ds, device_groups):
    multi_local_data = []
    for i in range(len(multi_ds)):
        ds = multi_ds[i]
        device_group = device_groups[i]
        device_index = get_device_index(device_group)
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
        multi_local_data.append(local_data)
    return multi_local_data

def get_device_index(device_group):
    local_device = hetu.local_device()
    if device_group.contains(local_device):
        device_index = device_group.get_index(local_device)
    else: # for pipeline parallel other stages
        device_index = -1 # only map placement group, will not map placement and do instantiate
    return device_index

# walkaround: just give order by type(placeholder/varibale), may not include all cases
def config2ds(config):
    num_devices = len(config['device_group'])
    zero = False
    split = {}
    for key, value in config['split'].items():
        split[int(key)] = value
    states = {-1: config['dup'], **split}
    if config['type'] == 'placeholder':
        order = [-1] + sorted(split.keys())
    elif config['type'] == 'variable':
        order = sorted(split.keys()) + [-1]
        assert 'zero' in config, f"variable config must have zero!"
        zero = config['zero']
    else:
        raise RuntimeError(f"unsupported type {config['type']}!")
    ds = hetu.DistributedStates(num_devices, states, order, zero)
    
    all_devices = hetu.global_device_group()
    device_group = hetu.DeviceGroup([all_devices.get(device_id) for device_id in config['device_group']])
    return ds, device_group

class HtMultiParallelLayerNorm(Module):
    def __init__(self, normalized_shape, multi_ds_parallel_config, eps=1e-5, dtype=hetu.float32, name='ln'):
        super(HtMultiParallelLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = [normalized_shape]  # type: ignore[assignment]
        self.normalized_shape = list(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.name = name
        self.ds_map = {'dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            self.ds_map['dup'].append(ds_dup)
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.ones_initializer()'), 
                                              self.normalized_shape, self.ds_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')
        self.bias = hetu.parallel_parameter(eval(f'hetu.zeros_initializer()'), 
                                              self.normalized_shape, self.ds_map['dup'], 
                                              self.device_index, dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_bias')

    def forward(self, input_p):
        return hetu.layer_norm(input_p, self.weight, self.bias, self.normalized_shape, self.eps, 
                               device_groups=self.device_groups, name=self.name)[0]

class HtMultiParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                 init_method='xavier_normal_', dtype=hetu.float32, name='embedding'):
        super(HtMultiParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name
        self.ds_map = {'dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_dup, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            self.ds_map['dup'].append(ds_dup)
        
        # embedding_table should not be splited in any dimension!
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_map['dup'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_groups=self.device_groups, name=f'{name}_table')
    
    def forward(self, input_p):
        return hetu.embedding_lookup(self.embedding_table, input_p, device_groups=self.device_groups, name=self.name)
    
class HtMultiVocabParallelEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, multi_ds_parallel_config, 
                init_method='xavier_normal_', dtype=hetu.float32, name='vocab_embedding'):
        super(HtMultiVocabParallelEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.name = name

        self.ds_map = {'split0_dup': [], 'dup_split0': []}
        self.device_index = []
        self.device_groups = []
        self.vocab_start_index = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_split0_dup, device_group = config2ds(ds_parallel_config) # for embedding table
            self.device_groups.append(device_group)
            dp, tp, num_devices = ds_parallel_config['dup'], ds_parallel_config['split'].get('0', 1), len(ds_parallel_config['device_group'])
            assert dp * tp == num_devices, f'VocabParallelEmbedding get wrong ds_parallel_config: {ds_parallel_config}!'
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [-1, 0]) # for data
            self.ds_map['split0_dup'].append(ds_split0_dup)
            self.ds_map['dup_split0'].append(ds_dup_split0)
            
            dup_group_idx = ds_split0_dup.get_dup_group_index(device_index)
            vocab_start_index = num_embeddings // tp * dup_group_idx
            self.vocab_start_index.append(vocab_start_index)

        # embedding_table was splited in vocab dimension
        self.embedding_table = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                                       [num_embeddings, embedding_dim], self.ds_map['split0_dup'], 
                                                       self.device_index, dtype=dtype, requires_grad=True, 
                                                       device_groups=self.device_groups, name=f'{name}_table')
    
    def forward(self, input_p):
        if input_p.check_multi_ds_equal(self.ds_map['dup_split0']):
            tensor_dup_split0 = input_p
        else:
            tensor_dup_split0 = hetu.comm(input_p, self.ds_map['dup_split0'])
            print(f"warning: vocab parallel embedding need extra communication for \
                    adapt input tensor distributed_states into {self.ds_map['dup_split0']}!")

        # walkaround: do offset inside embedding lookup op 
        # input_offset = tensor_split0_dup - self.vocab_start_index[0] # should do in embedding_lookup op for multi ds?
        lookup_partial_split0 = hetu.embedding_lookup(self.embedding_table, tensor_dup_split0, self.vocab_start_index, 
                                                      device_groups=self.device_groups, name=self.name+"_"+tensor_dup_split0.name)
        if lookup_partial_split0.check_multi_ds_equal(self.ds_map['dup_split0']): # pure dp
            output = lookup_partial_split0
        else:
            output = hetu.comm(lookup_partial_split0, self.ds_map['dup_split0'])
        return output

class HtMultiColumnParallelLinear(Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """
    def __init__(self, in_features, out_features, multi_ds_parallel_config,
                 bias=True, gather_output=True, init_method='xavier_normal_', 
                 dtype=hetu.float32, name='colp'):
        super(HtMultiColumnParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.name = name

        self.ds_map = {'dup_split0': [], 'split0_dup': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_split1_dup, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            dp, tp, num_devices, zero = ds_parallel_config['dup'], \
                                        ds_parallel_config['split'].get('1', 1), \
                                        len(ds_parallel_config['device_group']), \
                                        ds_parallel_config['zero']
            assert dp * tp == num_devices, f'ColumnParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'        
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            # when dp=1 tp=8, weights: ds_split0_dup->ds_split0, data: ds_dup_split0->ds_dup
            # when dp=8 tp=1, weights: ds_split0_dup->ds_dup, data: ds_dup_split0->ds_split0
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: dp, 0: tp}, [0, -1], zero) # for weights with trans_b
            ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [-1, 0]) # for data
            self.ds_map['dup_split0'].append(ds_dup_split0)
            self.ds_map['split0_dup'].append(ds_split0_dup)
        
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [out_features, in_features], 
                                              self.ds_map['split0_dup'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_map['split0_dup'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_bias')
        else:
            self.bias = None
      
    def forward(self, input_p):
        if input_p.check_multi_ds_equal(self.ds_map['dup_split0']):
            tensor_dup_split0 = input_p
        else:
            tensor_dup_split0 = hetu.comm(input_p, self.ds_map['dup_split0'])
            print(f"warning: column parallel linear need extra communication for \
                    adapt input tensor distributed_states {input_p.multi_distributed_states} into {self.ds_map['dup_split0']}!")
        
        tensor_split10 = hetu.linear(tensor_dup_split0, self.weight, self.bias, trans_b=True, device_groups=self.device_groups, name=self.name)
        if not self.gather_output:
            output = tensor_split10
        else:
            if tensor_split10.check_multi_ds_equal(self.ds_map['dup_split0']): # pure dp
                output = tensor_split10
            else:
                output = hetu.comm(tensor_split10, self.ds_map['dup_split0'])

        return output
    
# process: x->split1, w->split0 => y->partial => y->dup    
class HtMultiRowParallelLinear(Module):
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
    def __init__(self, in_features, out_features, 
                 multi_ds_parallel_config, bias=True, 
                 init_method='xavier_normal_', 
                 dtype=hetu.float32, name='rowp'):
        super(HtMultiRowParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.name = name
        
        self.ds_map = {'split0_dup': [], 'split1_dup': [], 'dup': [], 'split10': [], 'dup_split0': []}
        self.device_index = []
        self.device_groups = []
        for ds_parallel_config in multi_ds_parallel_config:
            ds_split0_dup, device_group = config2ds(ds_parallel_config)
            self.device_groups.append(device_group)
            dp, tp, num_devices, zero = ds_parallel_config['dup'], \
                                        ds_parallel_config['split'].get('0', 1), \
                                        len(ds_parallel_config['device_group']), \
                                        ds_parallel_config['zero']
            assert dp * tp == num_devices, f'RowParallelLinear get wrong ds_parallel_config: {ds_parallel_config}!'
            device_index = get_device_index(device_group)
            self.device_index.append(device_index)
            # assume num_devices=8, there exists 4 cases: dp=1 tp=8, dp=2 tp=4, dp=4 tp=2, dp=8 tp=1
            ds_split0_dup = hetu.DistributedStates(num_devices, {-1: dp, 0: tp}, [0, -1], zero) # for weight
            ds_split1_dup = hetu.DistributedStates(num_devices, {-1: dp, 1: tp}, [1, -1], zero) # for weight with trans_b
            ds_dup = hetu.DistributedStates(num_devices, {-1: num_devices}, [-1], zero) # for bias
            ds_split10 = hetu.DistributedStates(num_devices, {0: dp, 1: tp}, [1, 0]) # for data split in dimension 1
            ds_dup_split0 = hetu.DistributedStates(num_devices, {-1: tp, 0: dp}, [-1, 0]) # for data reduce partial to dup
            self.ds_map['split0_dup'].append(ds_split0_dup)
            self.ds_map['split1_dup'].append(ds_split1_dup)
            self.ds_map['dup'].append(ds_dup)
            self.ds_map['split10'].append(ds_split10)
            self.ds_map['dup_split0'].append(ds_dup_split0)
            
        self.weight = hetu.parallel_parameter(eval(f'hetu.{init_method}initializer()'), 
                                              [in_features, out_features], 
                                              self.ds_map['split0_dup'], self.device_index, 
                                              dtype=dtype, requires_grad=True, 
                                              device_groups=self.device_groups, name=f'{name}_weight')        
        if bias:
            self.bias = hetu.parallel_parameter(hetu.zeros_initializer(), [out_features], 
                                                self.ds_map['dup'], self.device_index,
                                                dtype=dtype, requires_grad=True, 
                                                device_groups=self.device_groups, name=f'{name}_bias')
        else:
            self.bias = None

    def forward(self, input_p):
        if input_p.check_multi_ds_equal(self.ds_map['split10']):
            tensor_split10 = input_p
        else:
            tensor_split10 = hetu.comm(input_p, self.ds_map['split10']) # exists src_ds == dst_ds case, just ignore it in comm_op

        tensor_partial_split0 = hetu.linear(tensor_split10, self.weight, trans_b=False, device_groups=self.device_groups, name=self.name)
        if tensor_partial_split0.check_multi_ds_equal(self.ds_map['dup_split0']): # pure dp
            tensor_dup_split0 = tensor_partial_split0
        else:
            tensor_dup_split0 = hetu.comm(tensor_partial_split0, self.ds_map['dup_split0'])
        output = tensor_dup_split0 + self.bias if self.bias is not None else tensor_dup_split0

        return output