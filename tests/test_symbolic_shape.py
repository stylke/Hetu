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

    init_shape = [2, 4, 9]
    _ = ht.parallel_placeholder(ht.float32, global_shape=init_shape, ds=ds_split01, device_group=all_device_group)
    _, _, x = ht.split(_, 3, 2)
    z = x + 3
    _, _, y = ht.split(z, 3, 2)

    x_shape = [2, 4, 3]
    x_data = np.zeros(x_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    # y_shape should be [1, 2, 1]
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("before:", y_data)
    
    x_shape = [4, 6, 6]
    x_data = np.ones(x_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    # y_shape should be [2, 3, 2]
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("after:", y_data)
        
    x_shape = [2, 4, 3]
    x_data = np.zeros(x_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    # y_shape should be [1, 2, 1]
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("again:", y_data)
        
    x_shape = [2, 4, 3]
    x_data = np.ones(x_shape)
    feed_dict = {x: parallel_data_provider(x_data, ds_split01, local_device_index)}
    results = y.graph.run(y, [y], feed_dict)
    # y_shape should be [1, 2, 1]
    y_data = results[0].numpy(force=True)
    
    if local_device_index == 0:
        print("again after again:", y_data)
    
if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()