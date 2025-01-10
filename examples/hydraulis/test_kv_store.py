import os
import socket
import argparse
import time
import numpy as np
import hetu as ht
from hetu.rpc.kv_store import KeyValueStoreClient, ProducerConsumer

local_device = None
all_devices = None

# Define a sample function to compute squares
def compute_square(n):
    return n * n

def distributed_init(args):
    global local_device, all_devices
    if 'HETU_LOCAL_HOSTNAME' not in os.environ:
        # 通过socket获取主机名并设置环境变量
        hostname = socket.gethostname()
        os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    else:
        print(f"Environment variable 'HETU_LOCAL_HOSTNAME' already set: {os.environ['HETU_LOCAL_HOSTNAME']}")
    ht.init_comm_group(args.ngpus, server_address = args.server_addr + ":" + args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

def test(args):
    
    nested_data = {
        "user1": {
            "profile": {
                "name": "Alice",
                "scores": np.array([95, 88, 92])
            },
            "metrics": np.array([0.5, 0.8])
        },
        "user2": {
            "profile": {
                "name": "Bob",
                "height": 666,
                "scores": np.array([75, 85, 80])
            },
            "metrics": np.array([0.7, 0.6])
        }
    }
    
    print("Initializing KeyValueStoreClient...")
    client = KeyValueStoreClient(address = args.server_addr + ":" + args.server_port)
    store_1 = client.register_dict('store 1')
    store_2 = client.register_dict('store 2')
    
    print("Putting key-value pairs...")
    if local_device.index == 0:
        store_1.put(key='key0', value={'x': np.ones((2, 2)) * 2, 'y': np.ones((4, 2)) * 6})
        store_2.put(key='key1', value=nested_data)
    if local_device.index == 1:
        store_2.put('key0', {'x': np.ones((2, 2)) * 10, 'y': np.ones((4, 2)) * 30})
        store_1.put('key1', {'x': np.ones((2, 2)) * 1, 'y': np.ones((4, 2)) * 3})
    ht.global_comm_barrier_rpc()
    print("Getting values from 'strategy plan'...")
    if local_device.index == 0:
        value = store_1.get('key1')
        print(f"{local_device}: get 'key1 from 'store_1' {value}")
        values = store_2.get_many(['key1', 'key0'])
        print(f"{local_device}: get 'key1' and 'key0' from 'store_2' {values}")
     
    print("Initializing ProducerConsumer...")   
    prod_cons = ProducerConsumer(client, max_workers=4)

    # Produce items with user-specified keys
    print("Producing items...")
    if local_device.index == 0:
        for i in range(5):
            key = f'item_{i}'
            prod_cons.produce(key, compute_square, i)
            time.sleep(0.2)  # Optional: Stagger the production a bit
    if local_device.index == 1:
        for i in range(5):
            key = f'item_{10 + i}'
            prod_cons.produce(key, compute_square, i)
            time.sleep(0.2)  # Optional: Stagger the production a bit
    
    # Consume items
    print("Consuming items...")
    for i in range(5):
        key = f'item_{i}'
        item = prod_cons.consume(key)
        print(f"Consumed {key}: {item}")
        key = f'item_{10 + i}'
        item = prod_cons.consume(key)
        print(f"Consumed {key}: {item}")
    
    # Shutdown the executor when done
    prod_cons.shutdown()


if __name__ == '__main__':
    print("Test kv store begin")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num of gpus"
    ) 
    parser.add_argument(
        "--seed", type=int, default=12345, help="random seed"
    ) 
    args = parser.parse_args()
    print("Hetu distributed init")
    distributed_init(args)
    print("Local device world rank is", all_devices.get_index(local_device))
    args.rank = all_devices.get_index(local_device)       
    test(args)
    print(f'{local_device}: test kv store end...')

