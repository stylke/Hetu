import os
import ptvsd
import socket
import hetu as ht

def distributed_init():
    hostname = socket.gethostname()
    # os.environ['HETU_LOCAL_HOSTNAME'] = os.environ['HOSTNAME']
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname

    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')
    # used for debug
    # ptvsd.enable_attach(address =('127.0.0.1', 4000 + all_devices.get_index(local_device)))
    # ptvsd.wait_for_attach()
    return local_device, all_devices