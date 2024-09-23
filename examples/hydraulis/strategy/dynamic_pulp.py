import pulp
import time
import json
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
from .cost_model import dynamic_strategy_time_cost

# 返回batch_seqlen_array中应该归属于当前strategy的indices
def dynamic_strategy(strategy_pool, strategy_id_list: List[int], max_seqlen_list: List[int], cur_strategy_relative_id: int, batch_seqlen_array: np.ndarray):
    # 先按照max_seqlen从大到小的顺序对strategy排序
    indices = list(range(len(strategy_id_list)))
    sorted_indices = sorted(indices, key=lambda x: max_seqlen_list[x], reverse=True)
    sorted_cur_strategy_relative_id = sorted_indices.index(cur_strategy_relative_id)
    sorted_strategy_id_list = [strategy_id_list[i] for i in sorted_indices]
    sorted_max_seqlen_list = [max_seqlen_list[i] for i in sorted_indices]
    print(f"Dynamic strategy, sorted_max_seqlen_list is {max_seqlen_list}, considering batch_seqlen_array {batch_seqlen_array.tolist()}")
    # 线性规划输入数据
    assert len(batch_seqlen_array.shape) == 1, "sorted_len shape must be [global_batch_size,]"
    B = len(batch_seqlen_array)  # 序列条数
    DP = len(sorted_strategy_id_list)  # 数据并行个数
    S = batch_seqlen_array  # 序列长度
    J = []  # 每个序列所能达到的max_seqlen最小的策略编号
    for i in range(B):
        max_j = None
        for j in range(DP):
            if S[i] <= sorted_max_seqlen_list[j]:
                max_j = j
            else:
                break
        assert max_j != None, f"can't find a strategy to run with seqlen {S[i]}"
        J.append(max_j)
        # print(f"seq {i} len is {batch_seqlen_array[i]}, and the max relative strategy id of it is {max_j}")
    
    # 定义最终优化的目标f
    def f(strategy_id, select_list, seqlen_list, max_seqlen=None):
        time_sum = 0
        for select, seqlen in zip(select_list, seqlen_list):
            time_sum += select * dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen) 
        if max_seqlen != None:
            time_sum += dynamic_strategy_time_cost(strategy_pool, strategy_id, max_seqlen) * (strategy_pool['strategies'][strategy_id]['pp'] - 1)
        return time_sum
    
    start_time = time.time()
    # 创建问题实例
    prob = pulp.LpProblem("ILP_Problem", pulp.LpMinimize)
    # 定义变量m_{i,j}
    m = pulp.LpVariable.dicts("m", (range(B), range(DP)), cat='Binary')
    # 定义辅助变量Z用于表示优化函数f的最大值
    Z = pulp.LpVariable("Z", lowBound=0)
    # 定义新的辅助变量Y_j表示当前j个数据并行组分到的最大的seq长度
    Y = pulp.LpVariable.dicts("Y", range(DP), lowBound=0)
    # 添加约束条件
    # sum(m_{i,j}) = 1 for all i
    for i in range(B):
        prob += pulp.lpSum(m[i][j] for j in range(J[i] + 1)) == 1
    # 添加约束条件
    # Y_j >= m_{i,j} * S_i for all i, j
    # 无法求解平方项的线性规划
    for j in range(DP):
        for i in range(B):
            prob += Y[j] >= m[i][j] * S[i]
    # 添加目标函数和相关约束
    # minimize Z and Z >= f(...)
    for j in range(DP):
        prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S, max_seqlen=Y[j])
        # prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        # prob += Z >= f(sorted_strategy_id_list[j], [m[i][j] for i in range(B)], S)
    # 设置目标函数
    prob += Z
    # 求解问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))
    end_time = time.time()

    # 输出结果
    # print(f"Status: {pulp.LpStatus[prob.status]}, time cost is {end_time - start_time}s")
    '''
    for i in range(B):
        for j in range(DP):
            print(f"m_{i}_{j} = {pulp.value(m[i][j])}")
    '''
    # print(f"Objective value: {pulp.value(prob.objective)}")
    for j in range(DP):
        pipeline_time_cost = f(sorted_strategy_id_list[j], [pulp.value(m[i][j]) for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        # print(f"Estimiated value of {j}-th dp is: {pipeline_time_cost}")
    
    cur_column = sorted_cur_strategy_relative_id
    res = []
    for i in range(B):
        if pulp.value(m[i][cur_column]):
            res.append(i)
    assert len(res) > 0, "currently not support zero seqs"
    # res = list(range(B // DP * sorted_cur_strategy_relative_id, B // DP * (sorted_cur_strategy_relative_id + 1)))
    return (pulp.value(prob.objective), res)

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
        prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) >= util_seqlen
        prob += pulp.lpSum(o[i, j] * seqs[i] for i in range(u)) <= max_seqlen
    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=5))
    if pulp.LpStatus[prob.status] == "Optimal":
        max_cost_value = pulp.value(max_cost)
        o_values = np.array([[pulp.value(o[i, j]) for j in range(v)] for i in range(u)])
        return max_cost_value, o_values
    else:
        # print(f"cannot put {seqs} into {v} micro batches and reach the full utilization")
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
    # print(f"micro-batching time cost is {end_time - start_time}s")
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
        raise RuntimeError("cannot gurantee the sequence ultilization for all DP")
        sys.exit(1)
    # print(f"Optimal e2e cost of the pipeline is {optimal_e2e_cost}")
    return (optimal_e2e_cost, optimal_o)

if __name__ == '__main__':
    # 读取并打印strategy数据
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
