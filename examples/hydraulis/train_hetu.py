import os
import signal
import time
import argparse
import socket
import pynvml
import ast
import json
import numpy as np
import hetu as ht
from hetu_llama import LLamaLMHeadModel
from llama_config import LLaMAConfig
from data_utils import LLaMAJsonDataset, build_data_loader
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_ds_parallel_config
from strategy import strategy_max_seqlen

local_device = None
all_devices = None
ds_parallel_config_path = "./ds_parallel_config/"

def distributed_init(args):
    global local_device, all_devices
    hostname = socket.gethostname()
    os.environ['HETU_LOCAL_HOSTNAME'] = hostname
    ht.init_comm_group(args.ngpus, server_address = args.server_addr + ":" + args.server_port)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

def train_dataset_provider(args):
    train_dataset = LLaMAJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.global_seq_len,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file
    )
    return train_dataset

def train_data_iterator(dataset, consumed_samples, global_batch_size):
    dataloader = build_data_loader(dataset, consumed_samples, global_batch_size)
    train_data_iter = iter(dataloader)
    return train_data_iter
  
def get_dg_from_union(device, dg_union):
    for i, dg in enumerate(dg_union):
        if dg.contains(device):
            return i, dg
    return None, None

def pretrain(args):
    
    # Generate & read configs
    with open(args.strategy_pool, 'r') as f:
        strategy_pool = json.load(f)
    multi_tp_pp_list = args.multi_tp_pp_list
    num_strategy = len(multi_tp_pp_list)
    multi_dp_size = [len(tp_pp_list) for tp_pp_list in multi_tp_pp_list]
    multi_gpu_pos = []
    multi_config_file_path = []
    multi_match_id_list = []
    multi_max_seqlen_list = []
    for strategy_id in range(num_strategy):
        # 获取当前异构dp策略下每个tp+pp子策略在pool中的id以及其支持的最大seq长度
        match_id_list = []
        max_seqlen_list = []
        for tp_pp in multi_tp_pp_list[strategy_id]:
            tp = tp_pp[0]
            pp = tp_pp[1]
            match_id = None
            for i, data in enumerate(strategy_pool['strategies']):
                if data['tp'] == tp and data['pp'] == pp:
                    match_id = i
                    break
            assert match_id != None, f"can't find tp{tp}pp{pp} in the strategy pool, please use the strategy within the pool"
            match_id_list.append(match_id)
            max_seqlen_list.append(strategy_max_seqlen(strategy_pool, match_id, multi_dp_size[strategy_id]))
        multi_match_id_list.append(match_id_list)
        multi_max_seqlen_list.append(max_seqlen_list)
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        layers_tp_groups, gpu_pos = convert_strategy(multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        generate_ds_parallel_config(args.ngpus, layers_tp_groups, config_file_path)
        multi_gpu_pos.append(gpu_pos)
        multi_config_file_path.append(config_file_path)
    
    ds_parallel_configs = read_ds_parallel_config(",".join(multi_config_file_path), num_strategy)
    config = LLaMAConfig(
        vocab_size=args.vocab_size, 
        n_embd=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        n_layer=args.num_hidden_layers, 
        n_head=args.num_attention_heads, 
        resid_pdrop=args.dropout_prob,
        embd_pdrop=args.dropout_prob,
        attn_pdrop=args.dropout_prob,
        activation_function=args.hidden_act,
        use_flash_attn=args.use_flash_attn
    )
    assert config.use_flash_attn == True, "symbolic shape can only used when flash attn is on for now"
    # Simple check for gpt blocks range
    ranges = []
    for _, block_config in ds_parallel_configs[0]['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][-1] == config.num_hidden_layers - 1, \
        f'gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!'

    # Hetu model definition
    model = LLamaLMHeadModel(config=config, ds_parallel_configs=ds_parallel_configs)

    input_ds_hierarchy, input_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'input')
    label_ds_hierarchy, label_dg_hierarchy = parse_multi_ds_parallel_config(ds_parallel_configs, 'label')
    # todo: remove the global_shape
    # now just offer a shape that can be divided by dp size
    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0]], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='input_ids')
    # position_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0]], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='position_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0]], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='token_type_ids')
    # attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[multi_dp_size[0]], ds_hierarchy=input_ds_hierarchy, device_group_hierarchy=input_dg_hierarchy, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[multi_dp_size[0]], ds_hierarchy=label_ds_hierarchy, device_group_hierarchy=label_dg_hierarchy, name='masked_lm_labels')
    config.cu_seqlens_list = []
    for block_id, block in enumerate(model.transformer.h):
        config.cu_seqlens_list.append(
            ht.parallel_placeholder(
                ht.int32, 
                global_shape=[multi_dp_size[0]], 
                ds_hierarchy=block.attn.qkv_dense.ds_union_map['split0_dup'], 
                device_group_hierarchy=block.attn.qkv_dense.device_group_unions,
                name=f'cu_seqlens_{block_id}'
            )
        )
    
    # 设置symbol
    # cp恒等于1
    config.multi_seq_lens_symbol = []
    config.multi_cp_group_symbol = []
    for i in range(len(input_ds_hierarchy)):
        assert multi_dp_size[i] == input_ds_hierarchy[i].get(0).get_dim(0), "dp size mismatches"
        # 例如[32, 32, 32, 48, 48, 32, 32, 32]
        # 表示各个dp分到的seq len
        # Hydraulis中mbs恒等于1
        # 其即是input_ids的shape 0
        config.multi_seq_lens_symbol.append([input_ids.symbolic_shape()[0] for _ in range(multi_dp_size[i])])
        # 例如[0, 0, 0, 1, 1, 2, 2, 2] 
        # 相同编号的在一个cp group中
        # Hydraulis中我们不使用cp
        config.multi_cp_group_symbol.append([ht.IntSymbol(i) for i in range(multi_dp_size[i])])
    # run plan时再根据当前GPU所在的tp pp组合来确定
    config.max_seqlen_symbol = ht.IntSymbol(1)

    print(f'{local_device}: build model begin...')
    loss = model(
        input_ids=input_ids,
        # position_ids=position_ids,
        # attention_mask=attention_mask,
        # token_type_ids=token_type_ids,
        labels=masked_lm_labels
    )
    print(f'{local_device}: build model end...')

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss)
    print(f'{local_device}: optimizer minimize end...')
    
    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')

    def run_plan(
        epoch = 0,
        strategy_id = 0
    ):     
        assert strategy_id < num_strategy, "strategy out of range"
        tp_pp_list = multi_tp_pp_list[strategy_id]
        max_seqlen_list = multi_max_seqlen_list[strategy_id]
        match_id_list = multi_match_id_list[strategy_id]
        gpu_pos = multi_gpu_pos[strategy_id]
        gpu_id = all_devices.get_index(local_device)
        dp_id, stage_id = None, None
        if gpu_id in gpu_pos:
            dp_id, stage_id = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id
        print(f"{local_device}: gpu_id={gpu_id}, dp_id={dp_id}, stage_id={stage_id}")

        # build dataloader and set max seqlen
        train_iter = None
        if dp_id != None:
            train_iter = train_data_iterator(train_dataset, consumed_samples, args.global_batch_size)
            config.max_seqlen_symbol.set_data(max_seqlen_list[dp_id])
            
        for step in range(args.steps):
            # load data for each dp
            packed_batch = None
            if train_iter:
                global_batch = next(train_iter)
                sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, train_dataset.pad_id())
                batch_indices = dynamic_strategy(strategy_pool, match_id_list, max_seqlen_list, dp_id, sorted_len)
                bucket = get_bucket(sorted_batch, train_dataset.pad_id(), batch_indices, max_seqlen_list[dp_id])
                packed_batch = bucket.packed_batch()
            if packed_batch == None or len(packed_batch) < 1: 
                raise NotImplementedError("currently not support GPUs with no data")
            else:
                packed_cu_seqlens_list = bucket.packed_cu_seqlens()
                labels_list = [packed_seq[1:].astype(np.int64) for packed_seq in packed_batch] # batch_size * [seq_len]
                tokens_list = [packed_seq[:-1].astype(np.int64) for packed_seq in packed_batch] # batch_size * [seq_len]
                # key : value = tensor : NDArrayList
                feed_dict = {
                    input_ids: tokens_list,
                    masked_lm_labels: labels_list
                }
                for i in range(config.n_layer):
                    feed_dict[config.cu_seqlens_list] = [x.astype(np.int32) for x in packed_cu_seqlens_list]
            start_time = time.time()
            try:
                results = train_op.graph.run(
                    loss, 
                    [loss, train_op], 
                    feed_dict = feed_dict, 
                    num_micro_batches = len(packed_batch), 
                    cur_strategy_id = strategy_id,
                )
            except RuntimeError as e:
                print(e)
                with open("./logs/exception.txt", 'w') as file:
                    print(f"device {gpu_id}:", file=file)
                    print(e, file=file)
                os.killpg(0, signal.SIGTERM)
            end_time = time.time()
            consumed_samples += args.global_batch_size
            # 如果在pipeline的最后一个stage上那么就打印loss
            if stage_id == tp_pp_list[dp_id][1] - 1:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        return consumed_samples
    
    # 运行
    def test(): 
        strategy_id = 0
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            consumed_samples = run_plan(epoch=epoch)
    
    test()

if __name__ == '__main__':
    print("Run hetu training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy_pool", type=str, default="./strategy/strategy_pool.json", help="json path to the strategy pool"
    )
    parser.add_argument(
        "--multi_tp_pp_list", type=str, default="[]", help="multi hetero dp strategy list"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="global training batch size"
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
        "--vocab_size", type=int, default=30522, help="total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="hidden size of transformer model",
    )
    parser.add_argument(
        "--ffn_hidden_size", type=int, default=-1, help="ffn hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="number of layers"
    )
    parser.add_argument(
        "--num_attention_heads", type=int, default=32, help="number of attention heads",
    )
    parser.add_argument(
        "--epochs", type=int, default=4, help="number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="use bfloat16."
    )
    parser.add_argument(
        "--server_addr", type=str, default='127.0.0.1', help="server's address"
    )
    parser.add_argument(
        "--server_port", type=str, default='23457', help="server's port"
    ) 
    parser.add_argument(
        "--ngpus", type=int, default=8, help="num of gpus"
    ) 
    args = parser.parse_args()
    print("Hetu distributed init")
    distributed_init(args)
    print("Local device world rank is", all_devices.get_index(local_device))
    args.multi_tp_pp_list = ast.literal_eval(args.multi_tp_pp_list)
    assert len(args.multi_tp_pp_list) >= 1, "there should be at least one strategy"
    with ht.graph("define_and_run", num_strategy=len(args.multi_tp_pp_list)):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            pretrain(args)
            print(f'{local_device}: train hetu ds parallel end...')
