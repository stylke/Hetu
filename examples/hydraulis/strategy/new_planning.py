import os
import resource
import pickle
import time
import fcntl
import concurrent.futures
import numpy as np
from typing import Callable, Any, Tuple, Dict
from .new_dynamic import dynamic_strategy, batching_strategy

def process_strategy(args):
    estimated_cost_1 = None
    batch_indices = None
    estimated_cost_2 = None
    batching_option_matrix = None
    
    (
        compute_strategy_id, multi_max_seqlen_list, 
        multi_match_id_list, strategy_pool, sorted_len
    ) = args
    
    max_seqlen_list = multi_max_seqlen_list[compute_strategy_id]
    match_id_list = multi_match_id_list[compute_strategy_id]
    
    # Call dynamic strategy (distributed)
    estimated_cost_1, all_dp_batch_indices = dynamic_strategy(strategy_pool, match_id_list, max_seqlen_list, sorted_len)
    print(f"strategy {compute_strategy_id}, estimated_cost_1 is {estimated_cost_1}, all_dp_batch_indices is {all_dp_batch_indices}")
    
    # hydraulis packing: balanced packing with utilization guaranteed
    all_dp_estimated_cost_2 = {}
    all_dp_batching_option_matrix = {}
    for dp_id, batch_indices in all_dp_batch_indices.items():
        if batch_indices is None:
            estimated_cost_2 = float('inf')
            batching_option_matrix = None
        else:
            # Call batching strategy (distributed)
            estimated_cost_2, batching_option_matrix = batching_strategy(
                strategy_pool, match_id_list[dp_id], 
                sorted_len[batch_indices], max_seqlen_list[dp_id]
            )
            # print(f"strategy {compute_strategy_id}, dp {dp_id}, estimated_cost_2 is {estimated_cost_2}")
            if not isinstance(batching_option_matrix, np.ndarray):
                print(f"{compute_strategy_id}-th strategy {dp_id}-th dp cannot guarantee the sequence utilization, the seqs that need to pack is {sorted_len[batch_indices]}")
        all_dp_estimated_cost_2[dp_id] = estimated_cost_2
        all_dp_batching_option_matrix[dp_id] = batching_option_matrix
        
    return estimated_cost_1, all_dp_batch_indices, all_dp_estimated_cost_2, all_dp_batching_option_matrix

def new_find_optimal_strategy(
    compute_strategy_id_list, multi_max_seqlen_list, 
    multi_match_id_list, strategy_pool, sorted_len, 
):
    # 先筛选出可以跑当前max_seqlen的strategy
    compute_strategy_id_list = [id for id in compute_strategy_id_list if max(multi_max_seqlen_list[id]) >= sorted_len[-1]]
    assert len(compute_strategy_id_list) > 0, f"no strategy can afford current max seqlen {sorted_len[-1]} in the global batch"
   
    # Prepare arguments for multiprocessing
    args_list = [
        (
            compute_strategy_id, multi_max_seqlen_list, 
            multi_match_id_list, strategy_pool, sorted_len
        )
        for compute_strategy_id in compute_strategy_id_list
    ]

    # print(f"Simutaneously handle strategies {compute_strategy_id_list}")
    
    results = []
    start_time = time.time()
    for args in args_list:
        result = process_strategy(args)
        results.append(result)
    end_time = time.time()
    print(f"Find optimal seqs-assigning & seqs-batching strategy time cost: {end_time - start_time}s")
    
    # Unpack the results
    estimated_cost_1_list = [res[0] for res in results]
    all_dp_batch_indices_list = [res[1] for res in results]
    all_dp_estimated_cost_2_list = [res[2] for res in results]
    all_dp_batching_option_matrix_list = [res[3] for res in results]
    
    # 依据estimated_cost_list中最小的值取出四个list对应的各个值
    min_cost_index = 0
    print(f"compute_strategy_id_list = {compute_strategy_id_list}, all_dp_estimated_cost_2_list = {all_dp_estimated_cost_2_list} , estimated_cost_1_list = {estimated_cost_1_list}")
    min_cost_index = np.argmin([max(all_dp_estimated_cost_2) for all_dp_estimated_cost_2 in all_dp_estimated_cost_2_list])
    
    optimal_estimated_cost_1 = estimated_cost_1_list[min_cost_index]
    optimal_all_dp_batch_indices = all_dp_batch_indices_list[min_cost_index]
    optimal_all_dp_estimated_cost_2 = all_dp_estimated_cost_2_list[min_cost_index]
    optimal_all_dp_batching_option_matrix = all_dp_batching_option_matrix_list[min_cost_index]
    
    value = {
        'optimal_compute_strategy_id': compute_strategy_id_list[min_cost_index],
        'optimal_estimated_cost_1': optimal_estimated_cost_1,
        'optimal_all_dp_batch_indices': optimal_all_dp_batch_indices,
        'optimal_all_dp_estimated_cost_2': optimal_all_dp_estimated_cost_2,
        'optimal_all_dp_batching_option_matrix': optimal_all_dp_batching_option_matrix
    }
    
    return value

