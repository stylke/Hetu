from tqdm import tqdm
import hetu as ht
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
    
    x = ht.parallel_parameter(eval(f'ht.xavier_normal_initializer()'), 
                                [12, 4], 
                                ds_split01, local_device_index, 
                                dtype=ht.float32, requires_grad=False, 
                                device_group=all_device_group, name='x')  
    print(x.get_device_group())
    print(x.get_data())
          
    x.reset_data(np.ones((6, 2)) * local_device_index)
    print(x.get_data())

    _ = x.graph.run(x, [x], {})
    print(x.get_data())
    
    
if __name__ == '__main__':
    with ht.graph("define_and_run"):
        unit_test()