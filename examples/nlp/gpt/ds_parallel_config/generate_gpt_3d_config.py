import argparse
import json
import os

def generate_gpt_3d_config(num_layers=32, num_gpus=8, dp=2, tp=2, pp=2, zero=True):
    num_layers_per_stage = num_layers // pp
    num_devices_per_stage = num_gpus // pp
    device_groups = [list(range(stage_id * num_devices_per_stage, (stage_id + 1) * num_devices_per_stage)) for stage_id in range(pp)]

    ds_parallel_config = {
        'zero': zero,
        'devices': list(range(num_gpus)),
        'input': {
            'split': {'0': dp},
            'dup': tp,
            'device_group': device_groups[0],
            'type': 'placeholder'
        },
        'gpt': {
            'wte': {
                'split': {'0': tp},
                'dup': dp,
                'device_group': device_groups[0],
                'type': 'variable'
            },
            'wpe': {
                'split': {},
                'dup': dp * tp,
                'device_group': device_groups[0],
                'type': 'variable'
            },
            'blocks': {

            },
            'layernorm_final': {
                'split': {},
                'dup': dp * tp,
                'device_group': device_groups[-1],
                'type': 'variable'
            }
        },
        'lm_head': {
            'split': {'1': tp},
            'dup': dp,
            'device_group': device_groups[-1],
            'type': 'variable'
        },
        'label': {
            'split': {'0': dp},
            'dup': tp,
            'device_group': device_groups[-1],
            'type': 'placeholder'
        }
    }

    for stage_id in range(pp):
        block_start_id = num_layers_per_stage * stage_id
        block_end_id = num_layers_per_stage * (stage_id + 1) - 1
        blocks_json = ds_parallel_config['gpt']['blocks']
        blocks_json[f'blocks{block_start_id}-{block_end_id}'] = {
            'range': [block_start_id, block_end_id],
            'layernorm1': {
                'split': {},
                'dup': dp * tp,
                'device_group': device_groups[stage_id],
                'type': 'variable'
            },
            'attn': {
                'qkv': {
                    'split': {'1': tp},
                    'dup': dp,
                    'device_group': device_groups[stage_id],
                    'type': 'variable'
                },
                'dense': {
                    'split': {'0': tp},
                    'dup': dp,
                    'device_group': device_groups[stage_id],
                    'type': 'variable'
                }
            },
            'layernorm2': {
                'split': {},
                'dup': dp * tp,
                'device_group': device_groups[stage_id],
                'type': 'variable'
            },
            'mlp': {
                'dense_h_to_4h': {
                    'split': {'1': tp},
                    'dup': dp,
                    'device_group': device_groups[stage_id],
                    'type': 'variable'
                },
                'dense_4h_to_h': {
                    'split': {'0': tp},
                    'dup': dp,
                    'device_group': device_groups[stage_id],
                    'type': 'variable'
                }
            }
        }
    return ds_parallel_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_layers', type=int, default=32, help='num layers of gpt.'
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
        '--zero', action='store_true', help='use zero or not.'
    )
    # parser.add_argument(
    #     '--save_folder', type=str, default='./'
    # )
    args = parser.parse_args()
    ds_parallel_config = generate_gpt_3d_config(args.num_layers, args.num_gpus, args.dp, args.tp, args.pp, args.zero)
    save_folder = f'./ds_parallel_config/gpus{args.num_gpus}'
    file_name = f'dp{args.dp}_tp{args.tp}_pp{args.pp}.json'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(f'{save_folder}/{file_name}', 'w') as f:
        json.dump(ds_parallel_config, f, indent=4)

