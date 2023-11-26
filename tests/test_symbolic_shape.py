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

    global_shape = [2, 4, 2]
    x = ht.parallel_placeholder(ht.float32, global_shape=global_shape, ds=ds_split01, device_group=all_device_group)
    y = x + 1

    global_shape = [2, 4, 2]
    x_data = np.zeros(global_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("before:", y_data)
    
    global_shape = [2, 6, 3]
    x_data = np.ones(global_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("after:", y_data)
    
if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()