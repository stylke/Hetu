import hetu
from hetu.nn.modules.parallel import parallel_data_provider
import numpy as np
import torch
import time
np.random.seed(2023)

ds_dup = hetu.DistributedStates(4, {-1: 4}, [-1])
ds_split0 = hetu.DistributedStates(4, {0: 4}, [0])
ds_split0_dup = hetu.DistributedStates(4, {-1: 2, 0: 2}, [0, -1])
ds_dup_split1 = hetu.DistributedStates(4, {-1: 2, 1: 2}, [-1, 1])
ds_split01 = hetu.DistributedStates(4, {0: 2, 1: 2}, [0, 1])

def static_run_tp_ds():
  local_device = hetu.local_device()
  all_devices = hetu.global_device_group()
  all_device_group = hetu.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
  local_device_index = all_device_group.get_index(local_device)
  devices_num = all_device_group.num_devices
  g = hetu.graph('define_and_run')
  with g:
    n = 8
    local_n = n // devices_num
    dim = 4
    x = hetu.placeholder(hetu.float32, [local_n, dim], ds=ds_split0, device_group=all_device_group, name='x')
    y = hetu.placeholder(hetu.float32, [local_n*2, dim], ds=ds_split0_dup, device_group=all_device_group, name='y')
    w = hetu.Tensor(np.ones((dim, dim)), dtype=hetu.float32, requires_grad=True, ds=ds_dup, device_group=all_device_group, name='w')
    w2 = hetu.Tensor(np.ones((dim, dim//2)), dtype=hetu.float32, requires_grad=True, ds=ds_dup_split1, device_group=all_device_group, name='w2')
    x2 = hetu.matmul(x, w, False, False, name='mm1')
    x3 = hetu.comm(x2, ds_split0_dup, name='comm_op1')
    x4 = hetu.matmul(x3, w2, False, False, name='mm2')
    x5 = hetu.sigmoid(x4, name='sigmoid')
    pred = hetu.comm(x5, ds_split0_dup, name='comm_op2')
    loss = hetu.binary_cross_entropy(pred, y, 'mean', name='bce_loss')
    optimizer = hetu.SGDOptimizer(0.1, 0.0)
    train_op = optimizer.minimize(loss)

    np.random.seed(2023 + local_device_index)
    data = np.random.normal(0, 1, (local_n, dim))
    labels = np.zeros((local_n*2, dim))

    ret = g.graph.run(loss, [loss, w, w2, train_op], feed_dict={x: data, y: labels})
    print(f'{local_device}: \nw_updated: {ret[1]}\nw2_updated: {ret[2]}')

def static_run_tp_ds_refactor():
  local_device = hetu.local_device()
  all_devices = hetu.global_device_group()
  all_device_group = hetu.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
  local_device_index = all_device_group.get_index(local_device)
  devices_num = all_device_group.num_devices
  g = hetu.graph('define_and_run')
  with g:
    n = 8
    dim = 4

    x = hetu.parallel_placeholder(hetu.float32, (n, dim), ds=ds_split0, device_group=all_device_group, name='x')
    y = hetu.parallel_placeholder(hetu.float32, (n, dim), ds=ds_split0_dup, device_group=all_device_group, name='y')
    w = hetu.parallel_parameter(hetu.xavier_normal_initializer(), (dim, dim), ds_dup, local_device_index, dtype=hetu.float32, 
                                requires_grad=True, device_group=all_device_group, name='w')
    w2 = hetu.parallel_parameter(hetu.xavier_normal_initializer(), (dim, dim), ds_dup_split1, local_device_index, dtype=hetu.float32, 
                                 requires_grad=True, device_group=all_device_group, name='w2')
    x2 = hetu.matmul(x, w, False, False, name='mm1')
    x3 = hetu.comm(x2, ds_split0_dup, name='comm_op1')
    x4 = hetu.matmul(x3, w2, False, False, name='mm2')
    x5 = hetu.sigmoid(x4, name='sigmoid')
    pred = hetu.comm(x5, ds_split0_dup, name='comm_op2')
    loss = hetu.binary_cross_entropy(pred, y, 'mean', name='bce_loss')
    optimizer = hetu.SGDOptimizer(0.1, 0.0)
    train_op = optimizer.minimize(loss)

    data = parallel_data_provider(np.random.normal(size=(n, dim)), ds_split0, local_device_index)
    labels = parallel_data_provider(np.zeros((n, dim)), ds_split0_dup, local_device_index)

    ret = g.graph.run(loss, [loss, w, w2, train_op], feed_dict={x: data, y: labels})
    print(f'{local_device}: w_updated: {ret[1]}; w2_updated: {ret[2]}')    
    
def test_row_parallel():
  local_device = hetu.local_device()
  all_devices = hetu.global_device_group()
  all_device_group = hetu.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
  local_device_index = all_device_group.get_index(local_device)
  devices_num = all_device_group.num_devices
  g = hetu.graph('define_and_run')
  with g:
    n = 4
    dim = 4
    x = hetu.parallel_placeholder(hetu.float32, (n, dim), ds=ds_dup, device_group=all_device_group, name='x')
    y = hetu.parallel_placeholder(hetu.float32, (n, dim * 2), ds=ds_dup, device_group=all_device_group, name='y')    

    row_parallel = hetu.nn.RowParallelLinear(dim, dim * 2, all_device_group)
    pred = row_parallel(x)
    pred = hetu.sigmoid(pred)
    loss = hetu.binary_cross_entropy(pred, y, 'mean', name='bce_loss')
    optimizer = hetu.SGDOptimizer(0.1, 0.0)
    train_op = optimizer.minimize(loss)

    data = parallel_data_provider(np.random.normal(size=(n, dim)), ds_dup, local_device_index)
    labels = parallel_data_provider(np.zeros((n, dim * 2)), ds_dup, local_device_index)

    ret = g.graph.run(loss, [loss, row_parallel.weight, row_parallel.bias, train_op], feed_dict={x: data, y: labels})
    print(f'{local_device}: w_updated: {ret[1]}, bias_updated: {ret[2]}')

def test_column_parallel():
  local_device = hetu.local_device()
  all_devices = hetu.global_device_group()
  all_device_group = hetu.DeviceGroup([all_devices.get(0), all_devices.get(1), all_devices.get(2), all_devices.get(3)])
  local_device_index = all_device_group.get_index(local_device)
  devices_num = all_device_group.num_devices
  g = hetu.graph('define_and_run')
  with g:
    n = 4
    dim = 4
    x = hetu.parallel_placeholder(hetu.float32, (n, dim), ds=ds_dup, device_group=all_device_group, name='x')
    y = hetu.parallel_placeholder(hetu.float32, (n, dim * 2), ds=ds_dup, device_group=all_device_group, name='y')      

    column_parallel = hetu.nn.ColumnParallelLinear(dim, dim * 2, all_device_group)
    pred = column_parallel(x)
    pred = hetu.sigmoid(pred)
    loss = hetu.binary_cross_entropy(pred, y, 'mean', name='bce_loss')
    optimizer = hetu.SGDOptimizer(0.1, 0.0)
    train_op = optimizer.minimize(loss)

    data = parallel_data_provider(np.random.normal(size=(n, dim)), ds_dup, local_device_index)
    labels = parallel_data_provider(np.zeros((n, dim * 2)), ds_dup, local_device_index)

    ret = g.graph.run(loss, [loss, column_parallel.weight, column_parallel.bias, train_op], feed_dict={x: data, y: labels})
    print(f'{local_device}: w_updated: {ret[1]}, bias_updated: {ret[2]}')    


if __name__ == '__main__':
  hetu.init_comm_group()
  # static_run_tp_ds_refactor()
  test_row_parallel()
  # test_column_parallel()
