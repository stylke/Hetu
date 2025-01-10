import os
import resource
import pickle
import time
import fcntl
import concurrent.futures
import numpy as np
from typing import Callable, Any, Tuple, Dict
from .dynamic_pulp import dynamic_strategy, batching_strategy

def increase_file_descriptor_limit():
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    # print(f"Current soft limit: {soft_limit}, hard limit: {hard_limit}")
    # 增加软限制，但不能超过硬限制
    new_soft_limit = min(65536, hard_limit) 
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
    # print(f"New soft limit: {new_soft_limit}")

# 在主程序开始时调用
# increase_file_descriptor_limit()
MAX_WORKERS = 1
func_call_folder = "./func_call"
tag = 0
strategy_cache = {}

# workaround
# hope to leverage grpc in the future
def distributed_call(func_call_folder, tag, need_max_cost: bool, distributed_status: Tuple[int, int, int, int, Dict[int, int]], func: Callable, *args: Any, **kwargs: Any):
    # synchronize all processes
    # but seems doesn't work
    # ht.global_comm_barrier_rpc() 
    start_time = time.time()
    strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu = distributed_status
    path = func_call_folder + f"/{func.__name__}_strategy{strategy_id}_dp{dp_id}.pkl"
    assert os.path.exists(func_call_folder), f"please ensure {func_call_folder} is created"
        # os.makedirs(func_call_folder)        
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, call func {func.__name__} begin...")
    if gpu_id == dp_representive_gpu[dp_id]:
        # representive rank process call the function and write the result to the file
        # print(f"call func {func.__name__} begin...")
        result = func(*args, **kwargs)
        with open(path, 'wb') as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                pickle.dump((result, tag), file)
                file.flush()
                os.fsync(file.fileno())
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
        # print(f"call func {func.__name__} end...")
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, call func {func.__name__} end...")
    # ht.global_comm_barrier_rpc() 
    all_dp_results = []
    dp_range = list(range(dp_size)) if need_max_cost else [dp_id,]
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, get dp range {dp_range} results of {func.__name__} begin...")
    for cur_id in dp_range:
        path = func_call_folder + f"/{func.__name__}_strategy{strategy_id}_dp{cur_id}.pkl"
        while True:
            try:
                with open(path, 'rb') as file:
                    try:
                        fcntl.flock(file, fcntl.LOCK_SH)
                        result, read_tag = pickle.load(file)
                    finally:
                        fcntl.flock(file, fcntl.LOCK_UN)
            except Exception as e:
                # print(f"Exception raise {e}")
                time.sleep(1)  # 等待文件写入完成
                continue
            if read_tag == tag:
                break
            else:
                # print(f"read tag = {read_tag} but cur func call tag = {tag}")
                time.sleep(1)  # 等待文件写入完成
        all_dp_results.append(result)
    # print(f"strategy {strategy_id}, dp {dp_id}, gpu {gpu_id}, get dp range {dp_range} results of {func.__name__} end...")
    # ht.global_comm_barrier_rpc() 
    end_time = time.time()
    if need_max_cost:
        max_cost_result = max(all_dp_results, key=lambda x: x[0])
        final_result = (max_cost_result[0], all_dp_results[dp_id][1])
    else:
        final_result = all_dp_results[0]
    # print(f"Distributed func call {func.__name__} time cost: {end_time - start_time}s")
    return final_result

def process_strategy(args):
    estimated_cost_1 = None
    batch_indices = None
    estimated_cost_2 = None
    batching_option_matrix = None
    
    (
        func_call_folder, tag,
        compute_strategy_id, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
        multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
        rank, batching_method, 
        strategy_pool, sorted_len
    ) = args
    
    dp_size = multi_dp_size[compute_strategy_id]
    tp_pp_list = multi_tp_pp_list[compute_strategy_id]
    max_seqlen_list = multi_max_seqlen_list[compute_strategy_id]
    match_id_list = multi_match_id_list[compute_strategy_id]
    gpu_pos = multi_gpu_pos[compute_strategy_id]
    dp_representive_gpu = multi_dp_representive_gpu[compute_strategy_id]
    gpu_id = rank
    
    assert gpu_id in gpu_pos, f"gpu {gpu_id} is not included in this training"
    dp_id, stage_id = gpu_pos[gpu_id].dp_id, gpu_pos[gpu_id].stage_id
    assert dp_id < dp_size, "dp size mismatches"
    
    # Call dynamic strategy (distributed)
    estimated_cost_1, batch_indices = distributed_call(
        func_call_folder, tag,
        True, (compute_strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu), 
        dynamic_strategy, strategy_pool, match_id_list, max_seqlen_list, dp_id, sorted_len
    )
    # print(f"strategy {compute_strategy_id}, dp {dp_id}, estimated_cost_1 is {estimated_cost_1}")
    
    # hydraulis packing: balanced packing with utilization guaranteed
    if batching_method == 4:
        if batch_indices is None:
            estimated_cost_2 = float('inf')
            batching_option_matrix = None
        else:
            # Call batching strategy (distributed)
            estimated_cost_2, batching_option_matrix = distributed_call(
                func_call_folder, tag,
                True, (compute_strategy_id, gpu_id, dp_id, dp_size, dp_representive_gpu), 
                batching_strategy, strategy_pool, match_id_list[dp_id], 
                sorted_len[batch_indices], max_seqlen_list[dp_id]
            )
            # print(f"strategy {compute_strategy_id}, dp {dp_id}, estimated_cost_2 is {estimated_cost_2}")
            if not isinstance(batching_option_matrix, np.ndarray):
                print(f"rank {rank}: {compute_strategy_id}-th strategy {dp_id}-th dp cannot guarantee the sequence utilization, the seqs that need to pack is {sorted_len[batch_indices]}")
    # greedy packing
    else:
        estimated_cost_2, batching_option_matrix = None, None
    
    return estimated_cost_1, batch_indices, estimated_cost_2, batching_option_matrix

def find_optimal_strategy(
    compute_strategy_id_list, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
    multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
    rank, batching_method, 
    strategy_pool, sorted_len, 
    save_strategy_cache=False, read_strategy_cache=False, strategy_cache_folder=None
):
    global func_call_folder, tag, strategy_cache
    
    # 先筛选出可以跑当前max_seqlen的strategy
    compute_strategy_id_list = [id for id in compute_strategy_id_list if max(multi_max_seqlen_list[id]) >= sorted_len[-1]]
    assert len(compute_strategy_id_list) > 0, f"no strategy can afford current max seqlen {sorted_len[-1]} in the global batch"
   
    # Prepare arguments for multiprocessing
    args_list = [
        (
            func_call_folder, tag,
            compute_strategy_id, multi_dp_size, multi_tp_pp_list, multi_max_seqlen_list, 
            multi_match_id_list, multi_gpu_pos, multi_dp_representive_gpu,
            rank, batching_method, 
            strategy_pool, sorted_len
        )
        for compute_strategy_id in compute_strategy_id_list
    ]

    # print(f"Simutaneously handle strategies {compute_strategy_id_list}")
    
    results = []
    skip_run = False
    start_time = time.time()
    if read_strategy_cache:
        with open(strategy_cache_folder + f"/{rank}_cache.pkl", 'rb') as f:
            strategy_cache = pickle.load(f)
        if tag in strategy_cache:
            results = strategy_cache[tag]
            skip_run = True
    if not skip_run:
        if MAX_WORKERS == 1:  
            for args in args_list:
                result = process_strategy(args)
                results.append(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(args_list), MAX_WORKERS)) as executor:
                results = list(executor.map(process_strategy, args_list))
    if save_strategy_cache:
        strategy_cache[tag] = results
        with open(strategy_cache_folder + f"/{rank}_cache.pkl", 'wb') as f:
            pickle.dump(strategy_cache, f)
    tag += 1
    end_time = time.time()
    print(f"Find optimal seqs-assigning & seqs-batching strategy time cost: {end_time - start_time}s")
    
    # Unpack the results
    estimated_cost_1_list = [res[0] for res in results]
    batch_indices_list = [res[1] for res in results]
    estimated_cost_2_list = [res[2] for res in results]
    batching_option_matrix_list = [res[3] for res in results]
    
    # 依据estimated_cost_list中最小的值取出四个list对应的各个值
    min_cost_index = None
    if any(estimated_cost_2_list):
        print(f"compute_strategy_id_list = {compute_strategy_id_list}, estimated_cost_2_list = {estimated_cost_2_list} , estimated_cost_1_list = {estimated_cost_1_list}")
        min_cost_index = np.argmin([cost for cost in estimated_cost_2_list if cost is not None])
    else:
        print(f"compute_strategy_id_list = {compute_strategy_id_list}, estimated_cost_1_list = {estimated_cost_1_list}")
        min_cost_index = np.argmin(estimated_cost_1_list)
    
    optimal_estimated_cost_1 = estimated_cost_1_list[min_cost_index]
    optimal_batch_indices = batch_indices_list[min_cost_index]
    optimal_estimated_cost_2 = estimated_cost_2_list[min_cost_index]
    optimal_batching_option_matrix = batching_option_matrix_list[min_cost_index]
    
    return (compute_strategy_id_list[min_cost_index], optimal_estimated_cost_1, optimal_batch_indices, 
            optimal_estimated_cost_2, optimal_batching_option_matrix)

