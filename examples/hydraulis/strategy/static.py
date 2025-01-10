import json
import time
import pickle
import re
from tqdm import tqdm
from cost_model import get_strategy_max_seqlen, static_strategy_time_cost

def dynamic_programming(data, counter, N, N_STEP, S, S_STEP, K, max_seqlen_list):
    # 初始化表t和trace
    # t[N][S]表示使用不超过N个GPU去处理数据集中所有不超过S的seqs
    t = [[float('inf')] * (S + 1) for _ in range(N + 1)]
    trace = [[None] * (S + 1) for _ in range(N + 1)]
    
    # 初始化边界条件
    for n in range(0, N + 1, N_STEP):
        t[n][0] = 0
        trace[n][0] = []
    
    # 动态规划求解
    for n in tqdm(range(N_STEP, N + 1, N_STEP), desc=f"Enumerating on GPUs number"):
        for s in range(S_STEP, S + 1, S_STEP):
            t[n][s] = t[n - N_STEP][s]  # 默认选择不使用额外的GPU
            trace[n][s] = trace[n - N_STEP][s]
            
            # 枚举策略k
            for k in range(K):
                tp = data['strategies'][k]['tp']
                pp = data['strategies'][k]['pp']
                strategy_num_gpus = tp * pp  # 策略k所需的GPU数量
                
                # 如果策略k不能支持当前序列长度，或者所需GPU超过当前GPU数量，跳过
                if max_seqlen_list[k] < s or strategy_num_gpus > n:
                    continue
                
                # 枚举区间长度l
                for l in range(S_STEP, s + 1, S_STEP):
                    # 枚举并行策略的数量d
                    for d in range(1, n // strategy_num_gpus + 1):  # d个策略并行处理
                        cost = static_strategy_time_cost(data, counter, k, s - l, s, S_STEP)
                        new_value = max(t[n - d * strategy_num_gpus][s - l], cost / d)
                        # 更新最优解
                        if new_value < t[n][s]:
                            t[n][s] = new_value
                            trace[n][s] = trace[n - d * strategy_num_gpus][s - l] + [(s - l, s, f"dp{d}tp{tp}pp{pp}")]  # 记录选择的策略和区间

    return t, trace

if __name__ == '__main__':
    
    # 7b
    file_path = 'strategy_pool_7b.json'
    N = 16  # 总的GPU数量
    S = 8192  # 数据集中最长的序列长度
    S_STEP = 128  # 序列长度的步长
    N_STEP = 1
    os_dp_tp_pp = (2, 4, 2)
    # 32b
    file_path = 'strategy_pool_32b.json'
    N = 1024  # 总的GPU数量
    S = 32768 + 256  # 数据集中最长的序列长度
    S_STEP = 1024  # 序列长度的步长
    N_STEP = 4
    os_dp_tp_pp = (64, 16, 1)
    '''
    # 13b
    file_path = 'strategy_pool_13b.json'
    N = 32  # 总的GPU数量
    S = 16384 + 128  # 数据集中最长的序列长度
    # S = 65536 + 128  # 数据集中最长的序列长度
    S_STEP = 128  # 序列长度的步长
    N_STEP = 1
    os_dp_tp_pp = (8, 4, 1)
    # 7b A800
    file_path = 'strategy_pool_7b.json'
    N = 32  # 总的GPU数量
    S = 32768 + 128  # 数据集中最长的序列长度
    S_STEP = 128  # 序列长度的步长
    N_STEP = 1
    os_dp_tp_pp = (8, 4, 1)
    '''
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Read strategy data:")
    print(json.dumps(data, indent=4))
    
    # 读取数据集的counter
    file_path = './examples/hydraulis/dataset_analysis/web_counter.pkl'
    # file_path = './examples/hydraulis/dataset_analysis/16k_code_counter.pkl'
    with open(file_path, 'rb') as f:
        counter = pickle.load(f)
    print("Read dataset counter")
    
    # 提前获取每个策略支持的最长seqlen
    max_seqlen_list = []
    K = len(data['strategies'])  # 策略数量
    for k in range(K):
        tp = data['strategies'][k]['tp']
        pp = data['strategies'][k]['pp']
        max_seqlen = get_strategy_max_seqlen(data, k, os_dp_tp_pp=os_dp_tp_pp)
        max_seqlen_list.append(max_seqlen)  # 策略k支持的最大序列长度
        print(f"tp{tp}pp{pp} max_seqlen is {max_seqlen}")
    
    # 调用动态规划函数，找到最优解
    begin_time = time.time()
    t, trace = dynamic_programming(data, counter, N, N_STEP, S, S_STEP, K, max_seqlen_list)
    end_time = time.time()
    print("time:", end_time - begin_time)
    
    # 打印最优解和最优方案
    multi_tp_pp_list = [[(os_dp_tp_pp[1], os_dp_tp_pp[2]) for _ in range(os_dp_tp_pp[0])]]
    for s in range(S_STEP, S + 1, S_STEP):
        # print(f"Optimal time cost: {t[N][s]}")
        # print(f"max_seqlen: {s}, optimal trace: {trace[N][s]}")
        if trace[N][s] is None:
            continue
        tp_pp_list = []
        for x in trace[N][s]:
            pattern = r'(dp|tp|pp)(\d+)'
            matches = re.findall(pattern, x[2])
            result = {key: int(value) for key, value in matches}
            for _ in range(result['dp']):
                tp_pp_list.append((result['tp'], result['pp']))
        if tp_pp_list not in multi_tp_pp_list:
            multi_tp_pp_list.append(tp_pp_list)
    sorted_multi_tp_pp_list = [sorted(sublist, key=lambda x: x[0], reverse=True) for sublist in multi_tp_pp_list]
    print(f"sorted_multi_tp_pp_list: {sorted_multi_tp_pp_list}")