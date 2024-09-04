import json
import pickle
from tqdm import tqdm
from cost_model import strategy_max_seqlen, static_strategy_time_cost

def dynamic_programming(data, counter, N, D, S, S_STEP, K):
    # 初始化表t和trace
    # t[N][D][S]表示使用不超过N个GPU并开DP=D去处理数据集中所有不超过S的seqs
    t = [[[float('inf')] * (S + 1) for _ in range(D + 1)] for _ in range(N + 1)]
    trace = [[[None] * (S + 1) for _ in range(D + 1)] for _ in range(N + 1)]
    
    # print("Dynamic programming initialize")
    for n in range(N + 1):
        t[n][0][0] = 0
        trace[n][0][0] = []
        for s in range(1, S + 1):
            for d in range(D + 1):
                t[n][d][s] = float('inf')
                trace[n][d][s] = []
    
    # print("Dynamic programming run")
    for n in tqdm(range(1, N + 1), desc=f"Enumerating on GPUs number"):
        for d in range(1, D + 1):
            for s in range(S_STEP, S + 1, S_STEP):
                t[n][d][s] = t[n - 1][d][s]
                trace[n][d][s] = trace[n - 1][d][s]
                # 注意这里策略编号是从1开始往上数
                # 实际要减去1
                for k in range(K):
                    tp = data['strategies'][k]['tp']
                    pp = data['strategies'][k]['pp']
                    strategy_num_gpus = tp * pp 
                    if strategy_max_seqlen(data, k, D) < s or strategy_num_gpus > n:
                        continue
                    for l in range(S_STEP, s + 1, S_STEP):
                        cost = static_strategy_time_cost(data, counter, k, s - l, s, S_STEP)
                        new_value = max(t[n - strategy_num_gpus][d - 1][s - l], cost)
                        # print(f"tp = {tp}, pp = {pp}, handle ({s - l}, {s}] seqs, time cost is {cost}")
                        if new_value < t[n][d][s]:
                            t[n][d][s] = new_value
                            trace[n][d][s] = trace[n - strategy_num_gpus][d - 1][s - l] + [(s - l, s, f"tp{tp}pp{pp}")] # 将[s-l, s]区间上的数据运行在策略k上
                    # print(f"GPUs = {n}, DP = {d}, handle (0, {s}] seqs, new time cost is {t[n][d][s]}")
    
    return t, trace

if __name__ == '__main__':
    # 读取并打印strategy数据
    file_path = 'strategy_pool.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print("Read strategy data:")
    print(json.dumps(data, indent=4))
    # 读取数据集的counter
    file_path = '../_counter.pkl'
    with open(file_path, 'rb') as f:
        counter = pickle.load(f)
    print("Read dataset counter")

    # 示例调用
    N = 16
    D_MAX = 4
    S = 8192
    S_STEP = 128
    K = 10
    
    best_t = float('inf')
    best_trace = None
    for D in range(1, D_MAX + 1):
        t, trace = dynamic_programming(data, counter, N, D, S, S_STEP, K)
        # 打印最优解和最优方案
        print(f"DP = {D}, optimal time cost:", t[N][D][S])
        print(f"DP = {D}, optimal trace:", trace[N][D][S])
        if t[N][D][S] < best_t:
            best_t = t[N][D][S]
            best_trace = trace[N][D][S]
    print("best time cost:", best_t)
    print("best trace:", best_trace)
