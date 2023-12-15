from tqdm import tqdm
import hetu as ht
import torch
import numpy as np
from hetu.nn.modules.parallel import parallel_data_provider
import ptvsd # used for debug

ds_split01 = ht.DistributedStates(4, {0: 2, 1: 2}, [0, 1])

ht.init_comm_group()
local_device = ht.local_device()
all_devices = ht.global_device_group()
all_device_group = ht.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
local_device_index = all_device_group.get_index(local_device)
devices_num = all_device_group.num_devices

def unit_test():

    x = ht.parallel_placeholder(ht.float32, global_shape=[2, 12, 5, 5], ds=ds_split01, device_group=all_device_group)
    z = ht.softmax(x, 3)

    x_data = torch.load("/home/gehao/lhy/log/softmax.pt").detach().numpy()

    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = z.graph.run(z, [z], feed_dict)
    z_data = results[0].numpy(force=True)
    
    pytorch_z = torch.nn.functional.softmax(torch.tensor(x_data), dim=-1)
    if local_device_index == 0:
        print("input:", x_data[0,3,...])
        print("hetu:", z_data[0,3,...])
        print("pytorch:", pytorch_z[0,3,...])
    
if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()