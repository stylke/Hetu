import pulp
import time
import json
import random
import numpy as np
from tqdm import tqdm
from functools import lru_cache
from pyscipopt import Model
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List
from .cost_model import dynamic_strategy_time_cost

MINI_TRIAL_NUM = 100

def process_trial(trial, batch_seqlen_array, strategy_pool, sorted_max_seqlen_list,
                  sorted_strategy_id_list):
    DP = len(sorted_strategy_id_list)
    original_indices = np.arange(len(batch_seqlen_array))
    best_accumulate_cost = float('inf')
    best_res = None
    pp_list = [strategy_pool['strategies'][sid]['pp'] for sid in sorted_strategy_id_list]
    
    # 缓存dynamic_strategy_time_cost的计算结果
    @lru_cache(maxsize=None)
    def cached_dynamic_strategy_time_cost(strategy_id, seqlen):
        return dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen)
        
    for mini_trial in range(trial, trial + MINI_TRIAL_NUM):
        random.seed(mini_trial)
        shuffled_indices = original_indices.copy()
        random.shuffle(shuffled_indices)
        shuffled_batch_seqlen_array = batch_seqlen_array[shuffled_indices]
        accumulate_cost = np.zeros(DP)
        pipeline_warmup_cooldown_cost = np.zeros(DP)
        total_cost = np.zeros(DP)
        res = {dp_id: [] for dp_id in range(DP)}
        # 初始分配：只能放在特定策略中的序列
        visited = set()
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            # 判断是否只能放在dp_id=0的策略中
            if len(sorted_max_seqlen_list) == 1 or (sorted_max_seqlen_list[0] >= seqlen and sorted_max_seqlen_list[1] < seqlen):
                strategy_id = sorted_strategy_id_list[0]
                cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
                accumulate_cost[0] += cost
                pp = pp_list[0]
                pipeline_warmup_cooldown_cost[0] = max(pipeline_warmup_cooldown_cost[0], cost * (pp - 1))
                total_cost[0] = accumulate_cost[0] + pipeline_warmup_cooldown_cost[0]
                res[0].append(shuffled_indices[shuffled_seq_id])
                visited.add(shuffled_seq_id)
        # 维护当前的最大累计开销
        max_total_cost = total_cost.max()
        # 分配其余的序列
        for shuffled_seq_id, seqlen in enumerate(shuffled_batch_seqlen_array):
            if shuffled_seq_id in visited:
                continue
            select_dp_id = None
            min_candidate_max_total_cost = float('inf')
            for dp_id in range(DP):
                if sorted_max_seqlen_list[dp_id] < seqlen:
                    continue  # 该策略无法容纳此序列长度
                strategy_id = sorted_strategy_id_list[dp_id]
                cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
                pp = pp_list[dp_id]
                new_accumulate_cost = accumulate_cost[dp_id] + cost
                new_pipeline_cost = max(pipeline_warmup_cooldown_cost[dp_id], cost * (pp - 1))
                new_total_cost_dp_id = new_accumulate_cost + new_pipeline_cost
                # 计算新的最大累计开销
                candidate_max_total_cost = max(max_total_cost, new_total_cost_dp_id)
                if candidate_max_total_cost < min_candidate_max_total_cost:
                    min_candidate_max_total_cost = candidate_max_total_cost
                    select_dp_id = dp_id
                if candidate_max_total_cost == min_candidate_max_total_cost and accumulate_cost[dp_id] == 0:
                    min_candidate_max_total_cost = candidate_max_total_cost
                    select_dp_id = dp_id
            assert select_dp_id is not None, f"Cannot select a proper dp to place a sequence with length {seqlen}"
            # 更新选定策略的累计开销和结果
            strategy_id = sorted_strategy_id_list[select_dp_id]
            cost = cached_dynamic_strategy_time_cost(strategy_id, seqlen)
            pp = pp_list[select_dp_id]
            accumulate_cost[select_dp_id] += cost
            pipeline_warmup_cooldown_cost[select_dp_id] = max(pipeline_warmup_cooldown_cost[select_dp_id], cost * (pp -1))
            total_cost[select_dp_id] = accumulate_cost[select_dp_id] + pipeline_warmup_cooldown_cost[select_dp_id]
            max_total_cost = max(max_total_cost, total_cost[select_dp_id])
            res[select_dp_id].append(shuffled_indices[shuffled_seq_id])
        # 检查并更新最优结果
        if max_total_cost < best_accumulate_cost:
            best_accumulate_cost = max_total_cost
            best_res = res.copy()
        # 检查是否有未分配序列的策略
        has_empty = any(len(best_res[dp_id]) == 0 for dp_id in range(DP))
        if has_empty:
            return (float('inf'), None)
    return (best_accumulate_cost, best_res)

# 返回batch_seqlen_array中应该归属于某strategy的indices
# 保证返回的indices按照从小到大的顺序排列
def dynamic_strategy(strategy_pool, strategy_id_list: List[int], max_seqlen_list: List[int], batch_seqlen_array: np.ndarray):
    # 先按照max_seqlen从大到小的顺序对strategy排序
    indices = list(range(len(strategy_id_list)))
    sorted_indices = sorted(indices, key=lambda x: max_seqlen_list[x], reverse=True)
    sorted_strategy_id_list = [strategy_id_list[i] for i in sorted_indices]
    sorted_max_seqlen_list = [max_seqlen_list[i] for i in sorted_indices]
    # print(f"Dynamic strategy, sorted_strategy_id_list is {sorted_strategy_id_list}, sorted_cur_strategy_relative_id is {sorted_cur_strategy_relative_id}, sorted_max_seqlen_list is {max_seqlen_list}, considering batch_seqlen_array {batch_seqlen_array.tolist()}")
    # 线性规划输入数据
    assert len(batch_seqlen_array.shape) == 1, "sorted_len shape must be [global_batch_size,]"

    best_accumulate_cost = float('inf')  # 记录最优的最大累积开销
    best_res = None  # 记录最优的分配结果
    start_time = time.time()
    best_accumulate_cost, best_res = process_trial(
        0,
        batch_seqlen_array,
        strategy_pool,
        sorted_max_seqlen_list,
        sorted_strategy_id_list,
    )
    end_time = time.time()
    print(f"assign seqs time cost is {end_time - start_time}s")
    if best_res is not None:
        best_res = {key: sorted(value) for key, value in best_res.items()}
    # 返回最优解
    return (best_accumulate_cost, best_res)

def solve_v_micro_batches(seqs: List[int], costs: List[float], max_seqlen: int, util_seqlen: int, v: int):
    u = len(seqs)
    # Create the LP problem
    prob = pulp.LpProblem("Minimize_Max_Cost", pulp.LpMinimize)
    # Decision variables
    o = pulp.LpVariable.dicts("o", ((i, j) for i in range(u) for j in range(v)), cat='Binary')
    # Objective function: minimize the maximum cost of any micro batch
    max_cost = pulp.LpVariable("max_cost", lowBound=0)
    for j in range(v):
        prob += pulp.lpSum(o[i, j] * costs[i] for i in range(u)) <= max_cost
    prob += max_cost
    # Constraints
    for i in range(u):
        prob += pulp.lpSum(o[i, j] for j in range(v)) == 1
    for j in range(v):
        # prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) >= util_seqlen
        prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) <= max_seqlen
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=0.5))
    if pulp.LpStatus[prob.status] in ["Optimal", "Integer Feasible", "Stopped"]:
        max_cost_value = pulp.value(max_cost)
        if max_cost_value is not None:
            o_values = np.array([[pulp.value(o[i, j]) for j in range(v)] for i in range(u)])
            return max_cost_value, o_values
        else:
            return float('inf'), None 
    else:
        # print(f"cannot put {seqs} into {v} micro batches and reach the full utilization")
        return float('inf'), None

def batching_strategy(strategy_pool, strategy_id: int, seqs: List[int], max_seqlen: int):
    strategy = strategy_pool['strategies'][strategy_id]
    tp = strategy['tp']
    pp = strategy['pp']
    util_seqlen = strategy_pool['cluster_config']['utilization_seqlen'][f'tp{tp}']
    total_tokens = sum(seqs)
    min_num_micro_batches = max(1, (total_tokens + max_seqlen - 1) // max_seqlen)
    max_num_micro_batches = min(len(seqs), total_tokens // util_seqlen)
    max_num_micro_batches = max(min_num_micro_batches, max_num_micro_batches)
    num_micro_batches_enum = range(min_num_micro_batches, min(max_num_micro_batches + 1, min_num_micro_batches + 16))
    costs = [dynamic_strategy_time_cost(strategy_pool, strategy_id, seq) for seq in seqs]
    results = []
    start_time = time.time()
    results = []
    for v in num_micro_batches_enum:
        result = solve_v_micro_batches(seqs, costs, max_seqlen, util_seqlen, v)
        results.append(result)
    end_time = time.time()
    print(f"micro-batching time cost is {end_time - start_time}s")
    optimal_e2e_cost = float('inf')
    optimal_o = None # optimal option (o_i_j means put i-th seq in j-th micro batch)
    optimal_v = None # optimal num micro batched
    for v, (max_cost_value, o_values) in zip(list(num_micro_batches_enum), results):
        if max_cost_value < float('inf'):
            e2e_cost = (max_cost_value + strategy['c']) * (pp - 1 + v)
            # print(f"form {v} micro batches will have e2e cost {e2e_cost}")
            if e2e_cost < optimal_e2e_cost:
                optimal_e2e_cost = e2e_cost
                optimal_o = o_values
                optimal_v = v
    if not isinstance(optimal_o, np.ndarray):
        # raise RuntimeError("cannot gurantee the sequence ultilization for all DP")
        # sys.exit(1)
        return float('inf'), None
    # print(f"Optimal e2e cost of the pipeline is {optimal_e2e_cost}")
    return optimal_e2e_cost, optimal_o

if __name__ == '__main__':
    # 读取并打印strategy数据
    file_path = 'strategy_pool_32b.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Read strategy data:")
    print(json.dumps(data, indent=4))
    
    # Example usage
    seqs = [1000, 2000, 3000, 4000, 5000]
    max_seqlen = 6000
    strategy_id = 1

    optimal_o = batching_strategy(data, strategy_id, seqs, max_seqlen)
    print("Optimal Assignment Matrix (o):", optimal_o)

