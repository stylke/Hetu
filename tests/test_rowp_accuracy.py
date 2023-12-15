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
        self.row = ht.nn.RowParallelLinear(
            nx, 
            nf,
            all_device_group,
            dp=2, # dp
            bias=False,
            # bias=False,
            name='row'
        )
        
    def forward(self, x):
        x = self.row(x)
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
        # self.bias = torch.nn.Parameter(torch.zeros(nf))
        torch.nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        bias = torch.zeros(self.nf)
        x = torch.addmm(bias, x.view(-1, x.size(-1)), self.weight)
        # x = torch.addmm(torch.zeros(size_out, dtype=torch.float32), x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
    
    
class pytorch_model(torch.nn.Module):

    def __init__(self, nf, nx):
        super().__init__()
        self.row = pytorch_linear(nf, nx)

    def forward(self, x):
        x = self.row(x)
        return x


def unit_test():
    
    nx = 768
    nf = 768
    
    input_data = torch.load("/home/gehao/lhy/log/x.pt").reshape(-1, nx).detach().numpy()
    row_weight = torch.load("/home/gehao/lhy/log/y.pt").detach().numpy().T
    if local_device_index == 0:
        print("input_data:", input_data[:5,:384], input_data[:5,:384].sum())
        print("row_weight:", row_weight[:,:384], row_weight[:,:384].sum())
    # input_data = np.random.rand(10, nx).astype(np.float32)
    # row_weight = np.random.rand(nf, nx).astype(np.float32)
    
    
    # hetu
    model = hetu_model(nf, nx)
    state_dict = {'row.weight': row_weight}
    model.load_state_dict(state_dict, local_device = local_device)
    input = ht.from_numpy_parallel(parallel_data_provider(input_data, ds_split01, local_device_index), 
                            ds=ds_split01, requires_grad=True, device_group=all_device_group, name='input')
    output = model(input) + 0
    feed_dict = {input: parallel_data_provider(input_data, ds_split01, local_device_index)}
    result = output.graph.run(output, [output], feed_dict = feed_dict)
    print(f"device = {local_device}, output shape = {result[0].numpy(force=True).shape}, output sum = {result[0].numpy(force=True).sum()}")

    # pytorch
    model = pytorch_model(nf, nx)
    state_dict = {'row.weight': torch.tensor(row_weight).T}
    model.load_state_dict(state_dict)
    input = torch.tensor(input_data, requires_grad=True)
    output = model(input)
    if local_device_index == 0:
        print(f"pytorch output shape = {output.shape}, output[:5,:] sum = {output[:5,:].sum()}")


if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()