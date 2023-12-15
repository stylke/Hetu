import hetu as ht
import numpy as np
import torch
import math 
from torch.optim import SGD

from hetu.nn.modules.parallel import parallel_data_provider

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

ds_dup = ht.DistributedStates(4, {-1: 4}, [-1])
ds_split0 = ht.DistributedStates(4, {0: 4}, [0])
ds_split0_dup = ht.DistributedStates(4, {-1: 2, 0: 2}, [0, -1])
ds_dup_split1 = ht.DistributedStates(4, {-1: 2, 1: 2}, [-1, 1])
ds_split01 = ht.DistributedStates(4, {0: 2, 1: 2}, [0, 1])


ht.init_comm_group()
local_device = ht.local_device()
all_devices = ht.global_device_group()
all_device_group = ht.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
local_device_index = all_device_group.get_index(local_device)
devices_num = all_device_group.num_devices


class hetu_model(ht.nn.Module):
    
    def __init__(self, nf, nx):
        super().__init__()
        self.col = ht.nn.ColumnParallelLinear(
            nf, # small 
            nx, # large
            all_device_group,
            dp=2,
            bias=True,
            # bias=False,
            gather_output=False,
            name='col'
            # skip_bias_add=True
        )
        self.av = ht.nn.NewGeLU()
        self.row = ht.nn.RowParallelLinear(
            nx, 
            nf,
            all_device_group,
            dp=2, # dp
            bias=True,
            # bias=False,
            name='row'
        )
        self.ln = ht.nn.ParallelLayerNorm(nf, all_device_group)
        
    def forward(self, x):
        x = self.col(x)
        # x = self.av(x)
        x = self.row(x)
        x = self.ln(x)
        return x
  
    
class pytorch_linear(torch.nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = torch.nn.Parameter(torch.empty(nx, nf))
        self.bias = torch.nn.Parameter(torch.zeros(nf))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x = torch.addmm(torch.zeros(size_out, dtype=torch.float32), x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    
    
class pytorch_model(torch.nn.Module):

    def __init__(self, nf, nx):
        super().__init__()
        self.col = pytorch_linear(nx, nf)
        self.row = pytorch_linear(nf, nx)
        self.ln = torch.nn.LayerNorm(nf)

    def forward(self, x):
        x = self.col(x)
        # x = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        x = self.row(x)
        x = self.ln(x)
        return x


def unit_test():
    
    nx = 1024
    nf = 512
    
    input_data = np.random.rand(4, nf).astype(np.float32)
    label_data = np.random.rand(4, nf).astype(np.float32)
    col_weight = np.random.rand(nx, nf).astype(np.float32)
    col_bias = np.random.rand(nx,).astype(np.float32)
    row_weight = np.random.rand(nf, nx).astype(np.float32)
    row_bias = np.random.rand(nf,).astype(np.float32)
    ln_weight = np.random.rand(nf,).astype(np.float32)
    ln_bias = np.random.rand(nf,).astype(np.float32)
    
    # hetu
    model = hetu_model(nf, nx)
    state_dict = {'col.weight': col_weight, 'col.bias': col_bias,
                  'row.weight': row_weight, 'row.bias': row_bias,
                  'ln.weight': ln_weight, 'ln.bias': ln_bias}
    model.load_state_dict(state_dict, local_device = local_device)
    input = ht.from_numpy_parallel(parallel_data_provider(input_data, ds_split0_dup, local_device_index), 
                            ds=ds_split0_dup, requires_grad=True, device_group=all_device_group, name='input')
    label = ht.parallel_placeholder(ht.float32, global_shape=[4, nf], ds=ds_split0_dup, device_group=all_device_group, name='label')
    output = model(input)
    loss = (output - label) * (output - label)
    opt = ht.SGDOptimizer(lr = 10, momentum = 0.0)
    train_op = opt.minimize(loss)
    feed_dict = {label: parallel_data_provider(label_data, ds_split0_dup, local_device_index)}
    result = train_op.graph.run(loss, [loss, input, train_op], feed_dict = feed_dict)
    print(f"device = {local_device}, loss sum = {result[0].numpy(force=True).sum()}, input sum = {result[1].numpy(force=True).sum()}, row weight after backward = {model.row.bias.get_data().sum()}")

    # pytorch
    model = pytorch_model(nf, nx)
    state_dict = {'col.weight': torch.tensor(col_weight).T, 'col.bias': torch.tensor(col_bias),
                  'row.weight': torch.tensor(row_weight).T, 'row.bias': torch.tensor(row_bias),
                  'ln.weight': torch.tensor(ln_weight), 'ln.bias': torch.tensor(ln_bias)}
    model.load_state_dict(state_dict)
    input = torch.tensor(input_data, requires_grad=True)
    label = torch.tensor(label_data)
    output = model(input)
    loss = (output - label) * (output - label)
    opt = SGD([input, model.col.weight, model.row.weight, model.row.bias, model.ln.weight], lr = 10)
    loss.backward(torch.ones_like(loss))
    opt.step()
    if local_device_index == 0:
        print(f"pytorch loss sum = {loss.sum()}, input sum = {input.sum()}, row weight after backward = {model.row.bias.sum()}")


if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()