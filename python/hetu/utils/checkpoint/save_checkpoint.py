import hetu as ht
import os
import torch
import numpy as np
from collections import OrderedDict


WEIGHTS_NAME = 'hetu_pytorch_model'
WEIGHTS_FORMAT = '.bin'


def retrieve_query_key_value_ordering(param, checkpoint_version, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_splits * num_heads * hidden_size, :]
    # for compatibility with later versions of NVIDIA Megatron-LM.
    # The inverse operation is performed inside Megatron-LM to read checkpoints:
    # https://github.com/NVIDIA/Megatron-LM/blob/v2.4/megatron/checkpointing.py#L209
    # If param is the weight tensor of the self-attention block, the returned tensor
    # will have to be transposed one more time to be read by HuggingFace GPT2.
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        saved_shape = (num_heads, hidden_size, num_splits) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        saved_shape = (num_heads, num_splits, hidden_size) + input_shape[1:]
        param = param.view(*saved_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def change_query_key_value_ordering(param, num_splits, num_heads, hidden_size):
    # Permutes layout of param tensor to [num_heads * num_splits * hidden_size, :]
    # The original layout of param tensor is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    original_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
    param = param.view(*original_shape)
    param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


# 请保证gpu有足够的显存进行allgather
def allgather_intra_group_param(param, value, local_device, idx):
    # print("allgather param:", param, "value:", value, "ds:", param.distributed_states)
        with ht.graph("define_and_run", create_new=True, prefix="save_" + param.name + str(idx)):
            device_group = param.get_device_group()
            local_value = ht.parallel_placeholder(param.dtype, global_shape=param.global_shape, 
                                                ds=param.distributed_states, device_group=device_group)
            num_devices = device_group.num_devices
            ds_dup = ht.DistributedStates(num_devices, {-1: num_devices}, [-1])
            global_value = ht.comm(local_value, ds_dup)
            # TODO: 目前无法fetch被替换的comm op
            bias = ht.parallel_parameter(eval(f'ht.zeros_initializer()'), 
                                                global_value.shape, ds_dup, device_group.get_index(local_device), 
                                                dtype=param.dtype, requires_grad=False, 
                                                device_group=device_group, name='zeros')    
            global_value = global_value + bias 
            feed_dict = {local_value: value}
            results = global_value.graph.run(global_value, [global_value], feed_dict=feed_dict)
            return results[0].numpy(force=True)
    


def save_state_dict(state_dict, checkpoint_file):
    
    return torch.save(state_dict, checkpoint_file)


# Save a PyTorch checkpoint
def save_checkpoint(model, optimizer, path, config=None, local_device=None):
    
    # with ht.graph("define_and_run", create_new=True, prefix="save_model"):
        # Numpy -> Tensor
    global_state_dict = OrderedDict()
    global_state_dict["model"] = OrderedDict()
    global_state_dict["optimizer"] = OrderedDict()
    all_device_groups = []
    local_state_dict = model.state_dict()
    for k in local_state_dict:
        param = model.state_dict(format='hetu')[k]
        device_group = param.get_device_group()
        
        if device_group not in all_device_groups:
            all_device_groups.append(device_group)
        # TODO: implement allgather_inter_group_param()
        if not device_group.contains(local_device):
            continue
        
        global_value = local_state_dict[k]
        if not param.distributed_states.is_pure_duplicate:
            # print(f"local device = {local_device}: trying to allgather {k}")
            global_value = allgather_intra_group_param(param, local_state_dict[k], local_device, k)
        global_state_dict["model"][k] = torch.tensor(global_value)
        # print(f"local device = {local_device}: finish {k}")
        
        # TODO: maybe implement it elsewhere
        # qkv_dense的weight和bias是[num_heads * 3 * hidden_size, :]
        # 要先转化成为原先的[3 * num_heads * hidden_size, :]才可以
        # 因为会在num_heads这一维度上进行distributed tensor的split切分
        if "qkv_dense" in k:
            assert config != None, "There should be a config when using qkv_dense."
            global_state_dict["model"][k] = retrieve_query_key_value_ordering(global_state_dict["model"][k], 
                                                            2.0,
                                                            3, 
                                                            config.num_attention_heads, 
                                                            config.hidden_size // config.num_attention_heads)
        
        opt_state_dict = optimizer.get_states(param)
        for opt_state_k, opt_state_param in opt_state_dict.items():
            state_key = k + "_" + opt_state_k
            # print(state_key)
            global_value = opt_state_param.get_data()
            if not opt_state_param.distributed_states.is_pure_duplicate:
                # print(f"local device = {local_device}: trying to allgather {k}")
                global_value = allgather_intra_group_param(param, global_value, local_device, state_key)
            global_state_dict["optimizer"][state_key] = torch.tensor(global_value)       

            if "qkv_dense" in state_key and "step" not in state_key:
                assert config != None, "There should be a config when using qkv_dense."
                global_state_dict["optimizer"][state_key] = retrieve_query_key_value_ordering(global_state_dict["optimizer"][state_key], 
                                                                2.0,
                                                                3, 
                                                                config.num_attention_heads, 
                                                                config.hidden_size // config.num_attention_heads)     
    
        
    # Time to save the checkpoint
    for i, device_group in enumerate(all_device_groups):
        # print(device_group) 
        if device_group.contains(local_device):
            if device_group.get_index(local_device) == 0:
                archive_file = WEIGHTS_NAME + f'-{i + 1}-of-{len(all_device_groups)}' + WEIGHTS_FORMAT
                archive_file = os.path.join(
                    path, archive_file
                )
                save_state_dict(global_state_dict, archive_file)
    # print("finish save checkpoint")
    
    