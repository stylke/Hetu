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
from data_utils import LLaMAJsonDataset, build_data_loader, get_sorted_batch_and_len, build_fake_batch_and_len, get_input_and_label_buckets
from parallel_utils import read_ds_parallel_config, parse_multi_ds_parallel_config, convert_strategy, generate_ds_parallel_config
from strategy import strategy_max_seqlen, dynamic_strategy, batching_strategy, distributed_call

local_device = None
all_devices = None
ds_parallel_config_path = "./ds_parallel_config/"
alignment = 128

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
        max_seq_len=args.max_seq_len,
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
            max_seqlen = strategy_max_seqlen(strategy_pool, match_id, multi_dp_size[strategy_id])
            aligned_max_seqlen = max_seqlen // alignment * alignment
            max_seqlen_list.append(aligned_max_seqlen)
        multi_match_id_list.append(match_id_list)
        multi_max_seqlen_list.append(max_seqlen_list)
        # 获取GPU的位置
        # 原则是不让tp跨机并尽可能贪心地让pp跨机
        layers_tp_groups, gpu_pos = convert_strategy(multi_tp_pp_list[strategy_id], args.ngpus, args.num_hidden_layers)
        config_file_path = ds_parallel_config_path + f"strategy_{strategy_id}.txt"
        generate_ds_parallel_config(args.ngpus, layers_tp_groups, config_file_path)
        print(f"Strategy {strategy_id}, gpu positions are: {gpu_pos}")
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
        config.multi_seq_lens_symbol.append([input_ids.symbolic_shape[0] for _ in range(multi_dp_size[i])])
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
    
    def hetu_train(
        feed_dict,
        num_micro_batches,
        strategy_id
    ):
        try:
            results = train_op.graph.run(
                loss, 
                [loss, train_op], 
                feed_dict=feed_dict, 
                num_micro_batches=num_micro_batches, 
                cur_strategy_id=strategy_id,
                run_level = ht.run_level("compute_only")
            )
        except RuntimeError as e:
            print(e)
            with open("./logs/exception.txt", 'w') as file:
                print(f"{local_device}:", file=file)
                print(e, file=file)
            os.killpg(0, signal.SIGTERM)
        return results

    def run_plan(
        epoch = 0,
        consumed_samples = 0,
        strategy_id = 0,
        warm_up = False,
        batching_method = 4, 
        max_padded_seqlen = None,
        fake_seqlens = []
    ):     
        # batching_method
        # 0 means padding
        # 1 means unblanced assigned packing (maybe not a proper baseline)
        # 2 means greedy packing with static shape
        # 3 means greedy packing with dynamic shape
        # 4 means hydraulis packing
        assert strategy_id < num_strategy, "strategy out of range"
        if max_padded_seqlen:
            max_padded_seqlen % alignment == 0, "max_padded_seqlen should be aligned"
        dp_size = multi_dp_size[strategy_id]
        tp_pp_list = multi_tp_pp_list[strategy_id]
        max_seqlen_list = multi_max_seqlen_list[strategy_id]
        match_id_list = multi_match_id_list[strategy_id]
        gpu_pos = multi_gpu_pos[strategy_id]
        gpu_id = all_devices.get_index(local_device)
        dp_id, stage_id = None, None
        if gpu_id in gpu_pos:
            dp_id, stage_id = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id
            assert dp_id < dp_size, "dp size mismatches"
        print(f"{local_device}: gpu_id = {gpu_id}, dp_id = {dp_id}, stage_id = {stage_id}")
        
        # 找到每个dp中编号最小的gpu_id
        # 后面需要用这些gpu代表去跑决策算法
        dp_representive_gpu = {}
        for cur_gpu_id, cur_pos in gpu_pos.items():
            if cur_pos.dp_id not in dp_representive_gpu:
                dp_representive_gpu[cur_pos.dp_id] = cur_gpu_id
            else:
                dp_representive_gpu[cur_pos.dp_id] = min(dp_representive_gpu[cur_pos.dp_id], cur_gpu_id)
        print("DP representive gpu:", dp_representive_gpu)

        # build dataloader and get sequence parallel degree
        train_iter = None
        # sequence_parallel_degree = None
        if dp_id != None:
            train_iter = train_data_iterator(train_dataset, consumed_samples, args.global_batch_size)
            # sequence_parallel_degree = tp_pp_list[dp_id][0] # sp = tp 
            
        if warm_up:
            print(f"{local_device}: warm up begin...")
            num_micro_batches = 8
            if dp_id != None:
                # packing
                if batching_method == 4:
                    max_seqlen = max_seqlen_list[dp_id]
                # padding (or original greedy packing with static or dynamic shape)
                else:
                    assert max_padded_seqlen, "you should provide the max seqlen when doing padding or static-shape packing"
                    max_seqlen = max_padded_seqlen
                assert max_seqlen % alignment == 0, "max seqlen should already be aligned"
                config.max_seqlen_symbol.set_data(max_seqlen)
                packed_cu_seqlens_list = [np.array([0, max_seqlen], dtype=np.int32)] * num_micro_batches
                input_list = [np.zeros((max_seqlen,), dtype=np.int64)] * num_micro_batches
                label_list = [np.zeros((max_seqlen,), dtype=np.int64)] * num_micro_batches
                feed_dict = {
                    input_ids: input_list,
                    masked_lm_labels: label_list
                }
                for i in range(config.n_layer):
                    feed_dict[config.cu_seqlens_list[i]] = packed_cu_seqlens_list
                hetu_train(feed_dict, num_micro_batches, strategy_id)
                print(f"{local_device}: warm up end")
                return
            else:
                raise NotImplementedError("currently not support GPUs with no data")
            
        for step in range(args.steps):
            # load data for each dp
            input_batch, label_batch, cu_seqlens_list = None, None, None
            if dp_id != None:
                if len(fake_seqlens) > 0:
                    sorted_batch, sorted_len = build_fake_batch_and_len(fake_seqlens, train_dataset.pad_id())
                else:
                    global_batch = next(train_iter).numpy()
                    sorted_batch, sorted_len = get_sorted_batch_and_len(global_batch, train_dataset.pad_id())
                # packing
                if batching_method > 0:
                    # unbalanced seqs assignment
                    if batching_method == 1:
                        assert args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size when padding"
                        batch_size_per_dp = args.global_batch_size // dp_size
                        batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                        batching_option_matrix = None
                    # balanced seqs assignment
                    if batching_method >= 2:
                        # batch_indices = dynamic_strategy(strategy_pool, match_id_list, max_seqlen_list, dp_id, sorted_len)
                        estimated_cost_1, batch_indices = distributed_call((gpu_id, dp_id, dp_representive_gpu), dynamic_strategy, strategy_pool, match_id_list, max_seqlen_list, dp_id, sorted_len)
                        # hydraulis packing: balanced packing with utilization guranteed
                        if batching_method == 4:
                            # batching_option_matrix = batching_strategy(strategy_pool, match_id_list[dp_id], sorted_len[batch_indices], max_seqlen_list[dp_id])
                            estimated_cost_2, batching_option_matrix = distributed_call((gpu_id, dp_id, dp_representive_gpu), batching_strategy, strategy_pool, match_id_list[dp_id], sorted_len[batch_indices], max_seqlen_list[dp_id]) 
                        # greedy packing
                        else:
                            estimated_cost_2, batching_option_matrix = None, None
                    # Question: 每个micro batch的实际的max_seqlen都不一样
                    # FlashAttn的这一属性的设置是否对性能有明显的影响有待探究
                    # 目前暂时将其设置成当前轮次所处理的最大的seqlen
                    config.max_seqlen_symbol.set_data(sorted_len[batch_indices[-1]] - 1) 
                    print(f"{local_device}: {dp_id}-th dp local batch indices is {batch_indices}, estimated cost is {estimated_cost_1}")
                    strategy_max_seqlen = max_seqlen_list[dp_id] 
                    static_shape = False
                    if batching_method == 2 or batching_method == 3:
                        assert max_padded_seqlen, "static-shape packing should provide the max seqlen after packing"
                        strategy_max_seqlen = max_padded_seqlen
                        static_shape = True
                    input_bucket, label_bucket = get_input_and_label_buckets(sorted_batch, train_dataset.pad_id(), batch_indices, strategy_max_seqlen, alignment)
                    input_bucket.pack_data(batching_option_matrix, static_shape)
                    label_bucket.pack_data(batching_option_matrix, static_shape)
                    input_batch, label_batch = input_bucket.packed_batch(), label_bucket.packed_batch()
                    cu_seqlens_list = input_bucket.packed_cu_seqlens_list()
                    print(f"{local_device}: {dp_id}-th dp seqlens after packed is {[len(seq) for seq in input_batch]}, estimated cost is {estimated_cost_2}")
                # padding
                if batching_method == 0:
                    assert args.global_batch_size % dp_size == 0, "global_batch_size should be divided by dp_size when padding"
                    batch_size_per_dp = args.global_batch_size // dp_size
                    batch_indices = list(range(batch_size_per_dp * dp_id, batch_size_per_dp * (dp_id + 1)))
                    assert max_padded_seqlen, "padding should provide the max seqlen after padding"
                    config.max_seqlen_symbol.set_data(max_padded_seqlen - 1) 
                    input_bucket, label_bucket = get_input_and_label_buckets(sorted_batch, train_dataset.pad_id(), batch_indices, max_padded_seqlen, alignment)
                    input_bucket.pad_data()
                    label_bucket.pad_data()
                    input_batch, label_batch = input_bucket.padded_batch(), label_bucket.padded_batch()
                    cu_seqlens_list = input_bucket.padded_cu_seqlens_list()
            if input_batch == None or len(input_batch) < 1: 
                raise NotImplementedError("currently not support GPUs with no data")
            else:
                input_list = [micro_batch.astype(np.int64) for micro_batch in input_batch] # batch_size * [seq_len]
                label_list = [micro_batch.astype(np.int64) for micro_batch in label_batch] # batch_size * [seq_len]
                # key : value = tensor : NDArrayList
                feed_dict = {
                    input_ids: input_list,
                    masked_lm_labels: label_list
                }
                for i in range(config.n_layer):
                    feed_dict[config.cu_seqlens_list[i]] = [x.astype(np.int32) for x in cu_seqlens_list]
            start_time = time.time()
            results = hetu_train(feed_dict, len(input_batch), strategy_id)
            end_time = time.time()
            consumed_samples += args.global_batch_size
            # 如果在pipeline的最后一个stage上那么就打印loss
            if stage_id == tp_pp_list[dp_id][1] - 1 and len(results) > 0:
                loss_out = results[0].numpy(force=True).mean()
                print(f"{local_device}: [Epoch {epoch}] (step {step}, consumed_samples = {consumed_samples}): loss = {loss_out:.3f}, time = {end_time - start_time:.4f}")
        return consumed_samples
    
    # 运行
    def test(): 
        run_plan(
            warm_up=True, 
            batching_method=args.batching_method,
            max_padded_seqlen=args.max_seq_len
        )
        strategy_id = 0
        for epoch in range(args.epochs):
            consumed_samples = 0 # should be reset when run next epoch
            consumed_samples = run_plan(
                epoch=epoch, 
                consumed_samples=consumed_samples,
                strategy_id=strategy_id,
                batching_method=args.batching_method,
                max_padded_seqlen=args.max_seq_len,
                fake_seqlens=ast.literal_eval(args.fake_seqlens)
            )
    
    test()

if __name__ == '__main__':
    print("Run hetu training")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake_seqlens", type=str, default="[]", help="seqlen list of fake data"
    )
    parser.add_argument(
        "--batching_method", type=int, default=4, help="batching method"
    )
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
        "--max_seq_len", type=int, default=4096, help="maximum sequence length in the whole dataset"
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
