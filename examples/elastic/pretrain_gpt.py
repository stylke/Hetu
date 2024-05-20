import os
import time
import argparse
import socket
import ast
import ptvsd
import hetu as ht
from elastic.models.gpt.gpt_model import GPTLMHeadModel
from elastic.models.gpt.gpt_config import GPTConfig
from elastic.engine.data_utils import GPTJsonDataset
from elastic.engine.parallel_config import read_ds_parallel_config
from elastic.engine.wrapper import ModelWrapper, OptimizerWrapper, DatasetWrapper
from elastic.engine.trainer import TrainerCtxs, Trainer

local_device = None
all_devices = None

def distributed_init(use_two_node: bool = False):
    
    if use_two_node:
        hostname = socket.gethostname()
        if hostname == 'job-26147b12-dd3f-4226-88a1-df64c6ec8ffa-master-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'A100-1'
        elif hostname == 'job-26147b12-dd3f-4226-88a1-df64c6ec8ffa-worker-0':
            os.environ['HETU_LOCAL_HOSTNAME'] = 'A100-2'
        else:
            raise ValueError(f"Unknown hostname: {hostname}")
    
    global local_device, all_devices
    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()

    # used for debug
    # ptvsd.enable_attach(address =('127.0.0.1', 4000 + all_devices.get_index(local_device)))
    # ptvsd.wait_for_attach()


def pretrain(args):
    
    # config
    ds_parallel_configs = read_ds_parallel_config(args)
    model_config = GPTConfig(
        vocab_size=args.vocab_size, 
        n_positions=args.seq_length,
        n_ctx=args.seq_length,
        n_embd=args.hidden_size,
        n_layer=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        seq_len=args.seq_length,
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        activation_function=args.hidden_act,
        global_batch_size=args.global_batch_size,
        use_flash_attn=args.use_flash_attn,
    )
    # build symbolic shape
    model_config.mbs_times_dp_symbol = ht.IntSymbol(0)
    model_config.seq_len_symbol = ht.IntSymbol(0)
    
    # simple check for blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == model_config.num_hidden_layers - 1, \
        f"blocks range: {ranges} is conflict with num_hidden_layers: {model_config.num_hidden_layers}!"

    # wrapper
    model_wrapper = ModelWrapper(GPTLMHeadModel, model_config) 
    optimizer_wrapper = OptimizerWrapper(ht.AdamOptimizer)
    dataset_wrapper = DatasetWrapper(GPTJsonDataset)
    
    # trainer
    trainer = Trainer(model_wrapper, optimizer_wrapper, dataset_wrapper)
    
    # build graph
    # ctxs should be profiled in advance
    hetero_tp_alpha = [1.0, 2.0, 4.0, 8.0]
    hetero_tp_weight = [1.0, 1.0, 1.0, 1.0]
    normal_compute_time = 4000.0
    memory_k = [2934, 2483, 2024, 1567]
    memory_d = [5939, 4558, 4474, 5527]
    memory_bound = 40536.0
    memory_safe_gap = 4096.0
    straggler_threshold = 1.2
    straggler_safe_gap = 0.3
    top_k = 3
    ctxs = TrainerCtxs(
        bf16=args.bf16,
        hetero_tp_alpha=hetero_tp_alpha,
        hetero_tp_weight=hetero_tp_weight,
        normal_layers=args.num_hidden_layers // args.pp,
        normal_mbn=args.global_batch_size // args.micro_batch_size // args.dp,
        normal_compute_time=normal_compute_time,
        memory_k=memory_k,
        memory_d=memory_d,
        memory_bound=memory_bound,
        memory_safe_gap=memory_safe_gap,
        straggler_threshold=straggler_threshold,
        straggler_safe_gap=straggler_safe_gap,
        top_k=top_k
    )
    trainer.build(args, ctxs, ds_parallel_configs)
    
    # begin training
    trainer.train(
        args, 
        ds_parallel_configs, 
        local_device,
        all_devices,
        strategy_id = 1 if args.hetero_pipeline else 0
    )
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--hetero_stage_gpus', type=int, default=2, help='num of gpus of a single stage'
    )
    parser.add_argument(
        "--hetero_pipeline", action="store_true", help="use heterogenous pipeline."
    )
    parser.add_argument(
        "--hetero_data", action="store_true", help="use heterogenous data for each heterogenous pipeline."
    )
    parser.add_argument(
        "--hetero_layers", type=str, help='hetero layers list.'
    )
    parser.add_argument(
        "--micro_batch_num_list", type=str, help='micro batch num list.'
    )
    parser.add_argument(
        "--rank_to_device_mapping", type=str, help='rank to device mapping.'
    )
    parser.add_argument(
        "--unused_rank", type=str, help='unused rank.'
    )
    parser.add_argument(
        "--run_straggler_experiment", action="store_true", help="run heterogenous pipeline experiment."
    )
    parser.add_argument(
        "--run_memory_experiment", action="store_true", help="run memory experiment."
    )
    parser.add_argument(
        "--use_two_node", action="store_true", help="use 2x8 gpus to run script."
    )
    parser.add_argument(
        "--switch", type=int, default=0, help='switch.'
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
    )
    parser.add_argument(
        "--straggler_file", default="", type=str, help="straggler experiment result file"
    )
    parser.add_argument(
        "--memory_file", default="", type=str, help="memory experiment result file"
    )
    parser.add_argument(
        "--num_strategy", type=int, default=1, help="multi ds num"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Training batch size each micro batch"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--dp", type=int, default=2, help="Number of dp",
    )
    parser.add_argument(
        "--tp", type=int, default=2, help="Number of tp"
    )
    parser.add_argument(
        "--pp", type=int, default=4, help="Number of pp",
    )
    parser.add_argument(
        "--zero", action="store_true", help="Use zero"
    )
    args = parser.parse_args()
    assert args.hetero_stage_gpus == args.tp, "arg mismatches"
    args.seq_len = args.seq_length
    args.rank_to_device_mapping = ast.literal_eval(args.rank_to_device_mapping)
    args.suspended_rank_list = ast.literal_eval(args.unused_rank)
    args.unused_rank_list = []
    args.hetero_layers = ast.literal_eval(args.hetero_layers)
    args.micro_batch_num_list = ast.literal_eval(args.micro_batch_num_list)
    args.hetero_micro_batch_num_list = args.micro_batch_num_list
    distributed_init(args.use_two_node)         
    pretrain(args)
    print(f'{local_device}: train hetu ds parallel end...')