import hetu
import os
import torch
import numpy as np
from collections import OrderedDict
import io


WEIGHTS_NAME = 'hetu_pytorch_model.bin'


def parallel_data_provider(global_data, ds, device_index):
    order, states = ds.order, ds.states
    local_map = hetu.map_to_local_data(ds, device_index)
    local_data = global_data.copy()
    dims = len(local_data.shape)
    begin_pos = [0] * dims
    out_shape = local_data.shape
    for dim in order:
        if dim < 0:
            continue
        splits = states[dim]
        split_index = local_map[dim]
        start = int(split_index * (global_data.shape[dim] / splits))
        stop = min(int((split_index + 1) * (global_data.shape[dim] / splits)), global_data.shape[dim])
        if isinstance(local_data, hetu.NDArray): 
            begin_pos[dim] = start
            out_shape[dim] = stop - start
        else:
            local_data = local_data.take(range(start, stop), axis=dim)
    if isinstance(local_data, hetu.NDArray):
        local_data = local_data.slice(begin_pos, out_shape)
    return local_data


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


def load_state_dict(checkpoint_file):
    
    return torch.load(checkpoint_file, map_location="cpu")


def load_checkpoint(model, optim, file, config=None, local_device=None):
    
    # Load from a PyTorch checkpoint
    # TODO: more than one file to load (if the model is quite big)
    # archive_file = os.path.join(
    #     path, WEIGHTS_NAME
    # )
    
    # Tensor -> Numpy
    state_dict = load_state_dict(file)
    for k in state_dict["model"]:
        # TODO: maybe implement it elsewhere
        # qkv_dense的weight和bias原先是[3 * num_heads * hidden_size, :]
        # 要先转化成为[num_heads * 3 * hidden_size, :]才可以
        # 因为会在num_heads这一维度上进行distributed tensor的split切分
        if "qkv_dense" in k and "step" not in k:
            assert config != None, "There should be a config when using qkv_dense."
            state_dict["model"][k] = change_query_key_value_ordering(state_dict["model"][k], 
                                                            3, 
                                                            config.num_attention_heads, 
                                                            config.hidden_size // config.num_attention_heads)
        state_dict["model"][k] = state_dict["model"][k].numpy()
        
    # Time to load the checkpoint
    model.load_state_dict(state_dict["model"], local_device=local_device)
    
    
    for name, param in model.named_parameters():
        device_group = param.get_device_group()
        assert(local_device is not None)
        if(device_group.contains(local_device) == False):
            continue
        device_index = device_group.get_index(local_device)
        for optim_state_key, opt_param in state_dict["optimizer"].items():
            if(opt_param.dtype == torch.bfloat16):
                opt_param = opt_param.to(torch.float32).numpy()
            else:
                opt_param = opt_param.numpy()           
            if(len(optim_state_key) > len(name) and optim_state_key[0:len(name)] == name):
                state_name = optim_state_key[len(name) + 1:] 
                if "step" in state_name:
                    assert [1] == list(opt_param.shape), "global shape mismatched!"
                else:
                    assert param.global_shape == list(opt_param.shape), "global shape mismatched!"
                data = parallel_data_provider(opt_param, param.distributed_states, device_index)
                optim.set_states(param, state_name, data)
    return model, optim


def to_hetu_format(tensor_dict):

    def get_name_and_params(destination, state_dict, prefix):
        if isinstance(state_dict, torch.Tensor):
            # print(state_dict)
            destination[prefix] = state_dict
        elif isinstance(state_dict, OrderedDict) or isinstance(state_dict, dict):
            for name, child in state_dict.items():
                get_name_and_params(destination, child, name if prefix == "" else prefix + "." + name)

    def fix_name(destination):
        state_dict = OrderedDict()

        fix_structure = {"language_model.":"",
                        "embedding.word_embeddings.weight":"transformer.wte.embedding_table",
                        "embedding.position_embeddings.weight":"transformer.wpe.embedding_table"}
        fix_name = {"encoder" : "transformer", "layers": "h", "self_attention" : "attn", "linear_proj" : "dense", 
                    "linear_qkv" : "qkv_dense", "final_layernorm":"ln_f", "input_norm":"ln_1",
                    "post_attention_norm":"ln_2", "query_key_value":"qkv_dense", "mlp":"mlp.parallel_mlp",
                    "final_norm": "ln_f"}
        for key, value in destination.items():
            for old, new in fix_structure.items():
                key = key.replace(old, new, 1)
            key_split = key.split(".")
            key_new = '.'.join([fix_name.get(item, item) for item in key_split])
            state_dict[key_new] = value
        state_dict["lm_head.weight"] = state_dict["transformer.wte.embedding_table"]
        return state_dict

    state_dict =  OrderedDict()
    state_dict["model"] = OrderedDict()
    state_dict["optimizer"] = OrderedDict()
    get_name_and_params(state_dict["model"], tensor_dict["model"], "")
    state_dict["model"] = fix_name(state_dict["model"])
    
    to_hetu_name = {"exp_avg": "mean", "exp_avg_sq" : "variance"}
    for idx, name in enumerate(state_dict["model"]): 
        if name == "lm_head.weight":
            for idx_, name_ in enumerate(state_dict["model"]): 
                if(name_ == "transformer.wte.embedding_table" and name == "lm_head.weight"):
                    for opt_name, param in tensor_dict["optimizer"]["optimizer"]["state"][idx_].items():
                        new_name = name + "_" + to_hetu_name[opt_name]
                        state_dict["optimizer"][new_name] = param    
                    new_name = name + "_" + "step"
                    state_dict["optimizer"][new_name] = torch.tensor([tensor_dict["optimizer"]["optimizer"]["param_groups"][0]["step"]], dtype=torch.int64)
        else:
            for opt_name, param in tensor_dict["optimizer"]["optimizer"]["state"][idx].items():
                new_name = name + "_" + to_hetu_name[opt_name]
                state_dict["optimizer"][new_name] = param
            new_name = name + "_" + "step"
            state_dict["optimizer"][new_name] = torch.tensor([tensor_dict["optimizer"]["optimizer"]["param_groups"][0]["step"]], dtype=torch.int64)
    return state_dict



# file存放每个gpu中分割后的参数
def load_checkpoint_from_megatron(model, optim, file, config=None, local_device=None):
    
    state_dict = to_hetu_format(load_state_dict(file))
    for k in state_dict["model"]:
        # TODO: maybe implement it elsewhere
        # qkv_dense的weight和bias原先是[3 * num_heads * hidden_size, :]
        # 要先转化成为[num_heads * 3 * hidden_size, :]才可以
        # 因为会在num_heads这一维度上进行distributed tensor的split切分
        if "qkv_dense" in k and "step" not in k:
            assert config != None, "There should be a config when using qkv_dense."
            state_dict["model"][k] = change_query_key_value_ordering(state_dict["model"][k], 
                                                            3, 
                                                            config.num_attention_heads, 
                                                            config.hidden_size // config.num_attention_heads)
        # print(k, state_dict["model"][k])
        if(state_dict["model"][k].dtype == torch.bfloat16):
            state_dict["model"][k] = state_dict["model"][k].to(torch.float32).numpy()
        else:
            state_dict["model"][k] = state_dict["model"][k].numpy()

    # Time to load the checkpoint
    model.load_state_dict(state_dict["model"], local_device=None)    
    
    for name, param in model.named_parameters():
        device_group = param.get_device_group()
        assert(local_device is not None)
        if(device_group.contains(local_device) == False):
            continue
        device_index = device_group.get_index(local_device)
        for optim_state_key, opt_param in state_dict["optimizer"].items():
            if(opt_param.dtype == torch.bfloat16):
                opt_param = opt_param.to(torch.float32).numpy()
            else:
                opt_param = opt_param.numpy()           
            if(len(optim_state_key) > len(name) and optim_state_key[0:len(name)] == name):
                state_name = optim_state_key[len(name) + 1:] 
                if "step" in state_name:
                    assert [1] == list(opt_param.shape), "global shape mismatched!"
                else:
                    assert param.shape == list(opt_param.shape), "global shape mismatched!"
                optim.set_states(param, state_name, opt_param)
    return model, optim