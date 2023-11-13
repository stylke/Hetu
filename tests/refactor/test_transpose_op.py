from tqdm import tqdm
import hetu as ht
import numpy as np
from hetu.nn.modules.parallel import parallel_data_provider
import ptvsd # used for debug

ds_dup = ht.DistributedStates(4, {-1: 4}, [-1])

ht.init_comm_group()
local_device = ht.local_device()
all_devices = ht.global_device_group()
all_device_group = ht.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
local_device_index = all_device_group.get_index(local_device)
devices_num = all_device_group.num_devices

def unit_test():

    size = [1, 5, 6, 64 * 3]
    x = ht.parallel_placeholder(ht.float32, global_shape=size, ds=ds_dup, device_group=all_device_group)
    a, b, c = ht.split(x, 3, x.ndim - 1)
    a = ht.contiguous(a)
    b = ht.contiguous(b)
    c = ht.contiguous(c)
    z = c.transpose([0, 2, 3, 1])

    x_data = np.random.rand(*size) * 100
    feed_dict = {x: x_data}
    results = z.graph.run(z, [z, c], feed_dict)
    z_data = results[0].numpy(force=True)
    y_data = results[1].numpy(force=True)
    y_numpy = x_data[...,64*2:]
    z_numpy = np.transpose(y_numpy, (0, 2, 3, 1))
    if local_device_index == 0:
        print(x_data[...,64*2:] - y_data)
        print(z_data - z_numpy)
    
if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()