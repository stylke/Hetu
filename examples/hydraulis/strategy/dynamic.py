import pulp
import numpy as np
from typing import List
from .cost_model import dynamic_strategy_time_cost

# 返回batch_seqlen_array中应该归属于当前strategy的indices
def dynamic_strategy(strategy_pool, strategy_id_list: List[int], max_seqlen_list: List[int], cur_strategy_id: int, batch_seqlen_array: np.ndarray):
    # 先按照max_seqlen从大到小的顺序对strategy排序
    indices = list(range(len(strategy_id_list)))
    sorted_indices = sorted(indices, key=lambda x: max_seqlen_list[x], reverse=True)
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
    
    # 定义最终优化的目标f
    def f(strategy_id, select_list, seqlen_list, max_seqlen=None):
        time_sum = 0
        for select, seqlen in zip(select_list, seqlen_list):
            time_sum += select * dynamic_strategy_time_cost(strategy_pool, strategy_id, seqlen) 
        if isinstance(max_seqlen, int):
            time_sum += dynamic_strategy_time_cost(strategy_pool, strategy_id, max_seqlen) * (strategy_pool['strategies'][strategy_id]['pp'] - 1)
        return time_sum
    
    # 创建问题实例
    prob = pulp.LpProblem("ILP_Problem", pulp.LpMinimize)
    # 定义变量m_{i,j}
    m = pulp.LpVariable.dicts("m", (range(B), range(DP)), cat='Binary')
    # 定义辅助变量Z用于表示优化函数f的最大值
    Z = pulp.LpVariable("Z", lowBound=0)
    # 定义新的辅助变量Y_j表示当前j个数据并行组分到的最大的seq长度
    # Y = pulp.LpVariable.dicts("Y", range(DP), lowBound=0)
    # 添加约束条件
    # sum(m_{i,j}) = 1 for all i
    for i in range(B):
        prob += pulp.lpSum(m[i][j] for j in range(J[i] + 1)) == 1
    # 添加约束条件
    # Y_j >= m_{i,j} * S_i for all i, j
    # 无法求解平方项的线性规划
    '''
    for j in range(DP):
        for i in range(B):
            prob += Y[j] >= m[i][j] * S[i]
    '''
    # 添加目标函数和相关约束
    # minimize Z and Z >= f(...)
    for j in range(DP):
        # prob += Z >= f(j, [m[i][j] for i in range(B)], S, max_seqlen=Y[j])
        # prob += Z >= f(j, [m[i][j] for i in range(B)], S, max_seqlen=sorted_max_seqlen_list[j])
        prob += Z >= f(j, [m[i][j] for i in range(B)], S)
    # 设置目标函数
    prob += Z
    # 求解问题
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # 输出结果
    print(f"Status: {pulp.LpStatus[prob.status]}")
    '''
    for i in range(B):
        for j in range(DP):
            print(f"m_{i}_{j} = {pulp.value(m[i][j])}")
    '''
    print(f"Objective value: {pulp.value(prob.objective)}")
    
    # 找到cur_strategy_id对应哪个并获得具体其要取哪些seqs
    cur_column = sorted_strategy_id_list.index(cur_strategy_id)
    res = []
    for i in range(B):
        if pulp.value(m[i][cur_column]):
            res.append(i)
    assert len(res) > 0, "currently not support zero seqs"
    return res