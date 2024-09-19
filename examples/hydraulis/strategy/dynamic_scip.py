import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
from pyscipopt import Model
from .cost_model import dynamic_strategy_time_cost

def dynamic_strategy(strategy_pool, strategy_id_list: List[int], max_seqlen_list: List[int], cur_strategy_relative_id: int, batch_seqlen_array: np.ndarray):
    indices = list(range(len(strategy_id_list)))
    sorted_indices = sorted(indices, key=lambda x: max_seqlen_list[x], reverse=True)
    sorted_cur_strategy_relative_id = sorted_indices.index(cur_strategy_relative_id)
    sorted_strategy_id_list = [strategy_id_list[i] for i in sorted_indices]
    sorted_max_seqlen_list = [max_seqlen_list[i] for i in sorted_indices]

    B = len(batch_seqlen_array)
    DP = len(sorted_strategy_id_list)
    S = batch_seqlen_array
    J = []

    for i in range(B):
        max_j = None
        for j in range(DP):
            if S[i] <= sorted_max_seqlen_list[j]:
                max_j = j
            else:
                break
        assert max_j is not None, f"can't find a strategy to run with seqlen {S[i]}"
        J.append(max_j)

    def f(strategy_id, select_list, seqlen_list, max_seqlen=None):
        time_sum = 0
        for select, seqlen in zip(select_list, seqlen_list):
            time_sum += select * dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen)
        if isinstance(max_seqlen, int):
            time_sum += dynamic_strategy_time_cost(strategy_pool, strategy_id, max_seqlen) * (strategy_pool['strategies'][strategy_id]['pp'] - 1)
        return time_sum

    start_time = time.time()
    model = Model("ILP_Problem")
    model.setParam('limits/time', 5.0)  # 设置求解时间限制为5秒
    model.setParam('display/verblevel', 0)  # 禁止显示求解过程中的消息
    m = {}
    
    # Binary decision variables
    for i in range(B):
        for j in range(J[i] + 1):
            m[i, j] = model.addVar(vtype="BINARY", name=f"m_{i}_{j}")

    Z = model.addVar(vtype="CONTINUOUS", name="Z", lb=0)

    # Constraints: sum(m_{i,j}) = 1 for all i
    for i in range(B):
        model.addCons(sum(m[i, j] for j in range(J[i] + 1)) == 1)

    # Add the objective function and related constraints
    for j in range(DP):
        model.addCons(Z >= f(sorted_strategy_id_list[j], [m[i, j] for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j]))

    # Set the objective
    model.setObjective(Z, "minimize")

    # Solve the problem
    model.optimize()
    end_time = time.time()

    print(f"Status: {model.getStatus()}, time cost is {end_time - start_time}s")
    print(f"Objective value: {model.getObjVal()}")

    for j in range(DP):
        pipeline_time_cost = f(sorted_strategy_id_list[j], [model.getVal(m[i, j]) for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        print(f"Estimated value of {j}-th dp is: {pipeline_time_cost}")

    cur_column = sorted_cur_strategy_relative_id
    res = []
    for i in range(B):
        if model.getVal(m[i, cur_column]) > 0.5:
            res.append(i)

    assert len(res) > 0, "currently not support zero seqs"
    return res


def solve_v_micro_batches(seqs: List[int], costs: List[float], max_seqlen: int, util_seqlen: int, v: int):
    u = len(seqs)
    model = Model("Minimize_Max_Cost")
    model.setParam('limits/time', 5.0)  # 设置求解时间限制为5秒
    model.setParam('display/verblevel', 0)  # 禁止显示求解过程中的消息

    o = {}
    for i in range(u):
        for j in range(v):
            o[i, j] = model.addVar(vtype="BINARY", name=f"o_{i}_{j}")

    max_cost = model.addVar(vtype="CONTINUOUS", name="max_cost", lb=0)

    for j in range(v):
        model.addCons(sum(o[i, j] * costs[i] for i in range(u)) <= max_cost)

    for i in range(u):
        model.addCons(sum(o[i, j] for j in range(v)) == 1)

    for j in range(v):
        model.addCons(sum(o[i, j] * seqs[i] for i in range(u)) >= util_seqlen)
        model.addCons(sum(o[i, j] * seqs[i] for i in range(u)) <= max_seqlen)

    model.setObjective(max_cost, "minimize")
    model.optimize()

    if model.getStatus() == "optimal":
        max_cost_value = model.getObjVal()
        o_values = np.array([[model.getVal(o[i, j]) for j in range(v)] for i in range(u)])
        return max_cost_value, o_values
    else:
        return float('inf'), None


def batching_strategy(strategy_pool, strategy_id: int, seqs: List[int], max_seqlen: int):
    strategy = strategy_pool['strategies'][strategy_id]
    tp = strategy['tp']
    pp = strategy['pp']
    util_seqlen = strategy_pool['cluster_config']['utilization_seqlen'][f'tp{tp}']
    total_tokens = sum(seqs)
    min_num_micro_batches = (total_tokens + max_seqlen - 1) // max_seqlen
    max_num_micro_batches = total_tokens // util_seqlen
    num_micro_batches_enum = range(min_num_micro_batches, max_num_micro_batches + 1)

    costs = [dynamic_strategy_time_cost(strategy_pool, strategy_id, seq) for seq in seqs]
    results = []

    start_time = time.time()
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(solve_v_micro_batches, seqs, costs, max_seqlen, util_seqlen, v) for v in num_micro_batches_enum]
        results = [future.result() for future in futures]
    end_time = time.time()

    print(f"Micro-batching time cost is {end_time - start_time}s")

    optimal_e2e_cost = float('inf')
    optimal_o = None
    optimal_v = None

    for v, (max_cost_value, o_values) in zip(list(num_micro_batches_enum), results):
        if max_cost_value < float('inf'):
            e2e_cost = (max_cost_value + strategy['c']) * (pp - 1 + v)
            if e2e_cost < optimal_e2e_cost:
                optimal_e2e_cost = e2e_cost
                optimal_o = o_values
                optimal_v = v

    assert isinstance(optimal_o, np.ndarray), "cannot guarantee the sequence utilization for all DP"
    print(f"Optimal e2e cost of the pipeline is {optimal_e2e_cost}")
    return optimal_o


if __name__ == '__main__':
    # 读取并打印 strategy 数据
    file_path = 'strategy_pool.json'
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
