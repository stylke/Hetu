import argparse
import json
import os

def generate_gpt_3d_config(hetero_layers, num_layers=32, num_gpus=8, dp=2, tp=2, pp=2, zero=True):
    if dp == 1:
        zero = False
    num_devices_per_stage = num_gpus // pp
    input_device_group = list(range(0, num_devices_per_stage))
    output_device_group = list(range(num_gpus - num_devices_per_stage, num_gpus))

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'input': {
            'split': {'0': dp},
            'dup': tp,
            'device_group': input_device_group,
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp},
                'dup': dp,
                'device_group': input_device_group,
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': dp * tp,
                'device_group': input_device_group,
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {},
                'dup': dp * tp,
                'device_group': output_device_group,
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp},
            'dup': dp,
            'device_group': output_device_group,
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp},
            'dup': tp,
            'device_group': output_device_group,
            'type': 'placeholder'
        }
    }
    
    for block_id in range(num_layers):
        hybrid_device_group = []
        for pipeline_id in range(dp):
            device_group_num = 0
            cnt = 0
            for hetero_layer in hetero_layers[pipeline_id]:
                cnt += hetero_layer
                if block_id < cnt:
                    break
                device_group_num += 1
            devices = list(range(device_group_num * num_devices_per_stage + tp * pipeline_id, 
                                 device_group_num * num_devices_per_stage + tp * (pipeline_id + 1)))
            hybrid_device_group += devices
        blocks_json = ds_parallel_config['gpt']['blocks']
        blocks_json[f'blocks{block_id}'] = {
            'range': [block_id,],
            'layernorm1': {
                'split': {},
                'dup': dp * tp,
                'device_group': hybrid_device_group,
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp},
                    'dup': dp,
                    'device_group': hybrid_device_group,
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp},
                    'dup': dp,
                    'device_group': hybrid_device_group,
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': dp * tp,
                'device_group': hybrid_device_group,
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp},
                    'dup': dp,
                    'device_group': hybrid_device_group,
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp},
                    'dup': dp,
                    'device_group': hybrid_device_group,
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_size', type=str, default='7b', help='size of gpt, 7b or 13b.'
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
        '--zero', action='store_true', help='use zero or not.'
    )
    # parser.add_argument(
    #     '--save_folder', type=str, default='./'
    # )
    args = parser.parse_args()
    if args.model_size == '7b':
        num_layers = 32
    elif args.model_size == '13b':
        num_layers = 40
    else:
        assert 'now only support 7b or 13b!'
        
    hetero_layers = args.hetero_layers.split(",")
    assert len(hetero_layers) == args.dp * args.pp, "size of heterogenous layers list should be equal to dp * pp"
    hetero_layers = [[int(hetero_layers[i * args.pp + j]) for j in range(args.pp)] for i in range(args.dp)]
    for pipeline in hetero_layers:
        assert sum(pipeline) == num_layers, "sum of heterogenous layers of a single pipeline should be equal to the num of total layers"
        
    assert args.dp * args.tp * args.pp == args.num_gpus, \
            f'dp * tp * pp = {args.dp * args.tp * args.pp} is not equal to num_gpus {args.num_gpus}!'
    ds_parallel_config = generate_gpt_3d_config(hetero_layers, num_layers, args.num_gpus, args.dp, args.tp, args.pp, args.zero)
    # save_folder = f'./hetero'
    save_folder = '/home/pkuhetu/lhy/multi_switch/examples/nlp/gpt/ds_parallel_config/hetero'
    file_name = f'dp{args.dp}_tp{args.tp}_pp{args.pp}.json'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

