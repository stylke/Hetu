import hetu as ht
import numpy as np
import torch
import os
import sys
import ptvsd
import argparse

# ptvsd.enable_attach(address =('127.0.0.1', 4000))
# ptvsd.wait_for_attach()

def attn(eager_device, seq_len, num_heads, head_dim, packing_num):
    if packing_num == 0:
        packing_args = (False, None, None, None, None)
        packing_num = 1 
    else:
        cu_seqlens_list = [0]
        for i in range(packing_num):
            cu_seqlens_list.append(i * seq_len)
        cu_seqlens_np = np.array(cu_seqlens_list, dtype=np.int32)
        cu_seqlens = ht.from_numpy(cu_seqlens_np)
        packing_args = (True, cu_seqlens, cu_seqlens, ht.IntSymbol(seq_len), ht.IntSymbol(seq_len))
    qkv_np = np.random.randn(seq_len * packing_num, num_heads * 3 * head_dim).astype(np.float32)  
    qkv = ht.from_numpy(qkv_np)
    qkv = ht.data_transfer(ht.bfloat16, qkv, eager_device) 
    # warm up
    for i in range(10):     
        attn_output = ht.parallel_attn(
            qkv,
            head_dim, 
            1, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
            [[ht.IntSymbol(seq_len * packing_num)]], 
            [[ht.IntSymbol(0)]],
            *packing_args
        )[0]
     
    time = 0
    profile_cnt = 1
    with ht.profiler(enabled = True, record_shapes = True) as profiler:   
        for i in range(profile_cnt): 
            attn_output = ht.parallel_attn(
                qkv,
                head_dim, 
                1, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
                [[ht.IntSymbol(seq_len * packing_num)]], 
                [[ht.IntSymbol(0)]],
                *packing_args
            )[0]
        op_types = len(profiler.summary()['optype_with_inputs_view'])
        assert op_types == 1, f"length mismatch, find {op_types} op types"
        for item in profiler.summary()['optype_with_inputs_view']:
            time = time + item[2]
    return time / profile_cnt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_file", type=str, default="", help='save file path.'
    )
    parser.add_argument(
        "--eager_device", type=str, default="cuda:7", help="eager device."
    )
    parser.add_argument(
        '--seq_len', type=int, default=1024, help='seq len.'
    )
    parser.add_argument(
        '--num_heads', type=int, default=32, help='num heads.'
    )
    parser.add_argument(
        '--head_dim', type=int, default=128, help='head dim.'
    )
    parser.add_argument(
        '--packing_num', type=int, default=0, help='packing num (unit is --seq_len).'
    )
    args = parser.parse_args()
    with ht.graph("eager"):
        with ht.context(eager_device=args.eager_device):
            time = attn(args.eager_device, args.seq_len, args.num_heads, args.head_dim, args.packing_num)
    if args.save_file == "":
        print(f"{time}s")
    else:
        with open(args.save_file, 'a') as file:
            file.write(f"{time}s\n")
    print("Done.")