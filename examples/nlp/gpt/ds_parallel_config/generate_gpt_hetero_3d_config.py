import argparse
import json
import os
import ast

def generate_gpt_3d_config(rank_to_device_mapping, unused_rank, hetero_layers, num_layers=32, num_gpus=8, dp=2, tp=2, pp=2, zero=True):
    if dp == 1:
        zero = False
    num_devices_per_stage = num_gpus // pp
    
    dp_union = [dp for _ in range(dp)]
    tp_union_list = []
    dg_union_list = []
    for block_id in range(num_layers):
        hybrid_tp_degree = []
        hybrid_device_group = []
        for pipeline_id in range(dp):
            device_group_num = 0
            cnt = 0
            for hetero_layer in hetero_layers[pipeline_id]:
                cnt += hetero_layer
                if block_id < cnt:
                    break
                device_group_num += 1
            devices = range(device_group_num * num_devices_per_stage + tp * pipeline_id, 
                            device_group_num * num_devices_per_stage + tp * (pipeline_id + 1))
            hybrid_tp_degree.append(len([device for device in devices if device not in unused_rank]))
            hybrid_device_group.append([rank_to_device_mapping[device] for device in devices if device not in unused_rank])
        tp_union_list.append(hybrid_tp_degree)
        dg_union_list.append(hybrid_device_group)

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'input': {
            'split': {'0': dp_union},
            'dup': tp_union_list[0],
            'device_group_union': dg_union_list[0],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp_union_list[0]},
                'dup': dp_union,
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': [tp_union_list[0][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[0],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {},
                'dup': [tp_union_list[-1][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[-1],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp_union_list[-1]},
            'dup': dp_union,
            'device_group_union': dg_union_list[-1],
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp_union},
            'dup': tp_union_list[-1],
            'device_group_union': dg_union_list[-1],
            'type': 'placeholder'
        }
    }
    
    for block_id in range(num_layers):
        blocks_json = ds_parallel_config['gpt']['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'layernorm1': {
                'split': {},
                'dup': [tp_union_list[block_id][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': [tp_union_list[block_id][i] * dp for i in range(dp)],
                'device_group_union': dg_union_list[block_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp_union_list[block_id]},
                    'dup': dp_union,
                    'device_group_union': dg_union_list[block_id],
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_layers', type=int, default=32, help='size of gpt, 7b is 32 and 13b is 40.'
    )
    parser.add_argument(
        '--num_gpus', type=int, default=8, help='num of gpus.'
    )
    parser.add_argument(
        '--dp', type=int, default=2, help='dp.'
    )
    parser.add_argument(
        '--tp', type=int, default=2, help='tp.'
    )
    parser.add_argument(
        '--pp', type=int, default=2, help='pp.'
    )
    parser.add_argument(
        '--hetero_layers', type=str, help='heterogenous layers list.'
    )
    parser.add_argument(
        '--rank_to_device_mapping', type=str, default="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}", help='device to rank mapping.'
    )
    parser.add_argument(
        '--unused_rank', type=str, default="[]", help='unused rank list.'
    )
    parser.add_argument(
        '--zero', action='store_true', help='use zero or not.'
    )
    parser.add_argument(
        '--file_name', type=str, default=""
    )
    args = parser.parse_args()
    num_layers = args.num_layers
        
    hetero_layers = args.hetero_layers.split(",")
    assert len(hetero_layers) == args.dp * args.pp, "size of heterogenous layers list should be equal to dp * pp"
    hetero_layers = [[int(hetero_layers[i * args.pp + j]) for j in range(args.pp)] for i in range(args.dp)]
    for pipeline in hetero_layers:
        assert sum(pipeline) == num_layers, "sum of heterogenous layers of a single pipeline should be equal to the num of total layers"
        
    assert args.dp * args.tp * args.pp == args.num_gpus, \
            f'dp * tp * pp = {args.dp * args.tp * args.pp} is not equal to num_gpus {args.num_gpus}!'
    ds_parallel_config = generate_gpt_3d_config(ast.literal_eval(args.rank_to_device_mapping), ast.literal_eval(args.unused_rank), hetero_layers, num_layers, args.num_gpus, args.dp, args.tp, args.pp, args.zero)
    save_folder = './ds_parallel_config/hetero'
    if args.file_name == "":
        file_name = f'dp{args.dp}_tp{args.tp}_pp{args.pp}.json'
    else:
        file_name = args.file_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

