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

setup_seed(15)

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
    def __init__(self):
        super().__init__()
        self.linear = ht.nn.RowParallelLinear(
            1024, # in
            128, # out
            all_device_group,
            2, # dp
            bias=True,
            name='linear'
        )
    def forward(self, x):
        return self.linear(x)
  
    
class pytorch_model(torch.nn.Module):
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
        # x = torch.addmm(torch.zeros(size_out), x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


def unit_test():
    
    input_data = np.random.rand(4, 1024).astype(np.float32)
    label_data = np.random.rand(4, 128).astype(np.float32)
    linear_weight = np.random.rand(128, 1024).astype(np.float32)
    linear_bias = np.random.rand(128,).astype(np.float32)
    
    # hetu
    model = hetu_model()
    state_dict = {'linear.weight': linear_weight, 'linear.bias': linear_bias}
    # state_dict = {'linear.weight': linear_weight}
    model.load_state_dict(state_dict, local_device = local_device)
    input = ht.from_numpy_parallel(parallel_data_provider(input_data, ds_split01, local_device_index), 
                            ds=ds_split01, requires_grad=True, device_group=all_device_group, name='input')
    label = ht.parallel_placeholder(ht.float32, global_shape=[4, 128], ds=ds_split0_dup, device_group=all_device_group, name='label')
    # av = ht.nn.NewGeLU()
    # output = av(input)
    output = model(input)
    loss = (output - label)
    opt = ht.SGDOptimizer(lr = 10, momentum = 0.0)
    train_op = opt.minimize(loss)
    feed_dict = {label: parallel_data_provider(label_data, ds_split0_dup, local_device_index)}
    result = train_op.graph.run(loss, [loss, model.linear.weight, train_op], feed_dict = feed_dict)
    print(f"device = {local_device}, loss sum = {result[0].numpy(force=True).sum()}, weight after backward = {model.linear.weight.get_data().sum()}")

    # pytorch
    model = pytorch_model(128, 1024)
    state_dict = {'weight': torch.tensor(linear_weight).T, 'bias':  torch.tensor(linear_bias)}
    # state_dict = {'weight': torch.tensor(linear_weight).T}
    model.load_state_dict(state_dict)
    input = torch.tensor(input_data, requires_grad=True)
    label = torch.tensor(label_data)
    # output = 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    output = model(input)
    loss = (output - label)
    opt = SGD([input, model.weight], lr = 10)
    loss.backward(torch.ones_like(loss))
    opt.step()
    if local_device_index == 0:
        # print(f"init weight = {linear_weight}")
        print(f"pytorch loss sum = {loss.sum()}, weight after backward = {model.weight.sum()}")


if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()