import json
import numpy as np

# key是粗策略编号与seqs的span起始下标组成的tuple
# value是其所消耗的时间以及span内最大的seq长度
cache = {}

def linear_predict(s, b, c):
    return b * s + c

def quadratic_predict(s, a, b, c):
    return a * (s ** 2) + b * s + c

def dynamic_strategy_time_cost(data, strategy_id, s):
    strategy = data['strategies'][strategy_id]
    if isinstance(s, (int, np.int32, np.int64)):
        return quadratic_predict(s, strategy['a'], strategy['b'], 0)
    else:
        return linear_predict(s, strategy['b'], strategy['c'])

# 策略编号为strategy_id数据并行维度为D的情况下处理数据集中长度在[s - l, s]区间范围内的所消耗的时间
def static_strategy_time_cost(data, counter, strategy_id, s_begin, s_end, S_STEP):
    strategy = data['strategies'][strategy_id]
    sum_time = 0
    max_seq_len = 0
    # 左开右闭
    for span_begin in range(s_begin + 1, s_end + 1, S_STEP):
        cache_key = (strategy_id, span_begin)
        if cache_key in cache:
            sum_time += cache[cache_key][0]
            max_seq_len = max(max_seq_len, cache[cache_key][1]) # 后者可能是0表示span为空
            continue
        span_sum_time = 0
        span_max_seq_len = 0
        for s in range(span_begin, span_begin + S_STEP):
            if s in counter:
                # 1F1B
                span_sum_time += quadratic_predict(s, strategy['a'], strategy['b'], strategy['c']) * counter[s]
                span_max_seq_len = s
        cache[cache_key] = (span_sum_time, span_max_seq_len)
        sum_time += span_sum_time
        max_seq_len = max(max_seq_len, span_max_seq_len)
    # warm-up + cool-down
    sum_time += quadratic_predict(max_seq_len, strategy['a'], strategy['b'], strategy['c']) * (strategy['pp'] - 1)
    return sum_time

# 策略编号为strategy_id数据并行维度为D的情况下所能支持的最长seqlen
def strategy_max_seqlen(data, strategy_id, D, unit_test=False):
    gpus_per_node = data['cluster_config']['gpus_per_node']
    gpu_memory_bound = data['cluster_config']['gpu_memory_bound']
    tp = data['strategies'][strategy_id]['tp']
    pp = data['strategies'][strategy_id]['pp']
    A = data['memory_regression']['A'][f'tp{tp}']
    B = data['memory_regression']['B']
    alpha = data['memory_regression']['alpha']
    # workaround: significant gap occurs on zhiyuan A100
    # gap = data['memory_regression']['gap']['node' if tp * pp * D >= gpus_per_node else f'gpu{tp * pp * D}']
    gap = data['memory_regression']['gap']['node']
    safe_bound = data['memory_regression']['safe_bound']
    L = data['model_config']['L']
    H = data['model_config']['H']
    V = data['model_config']['V'] 
    p_g_os_memory = (L / pp) * (H * H / tp) * B * (alpha / D + 1 - alpha) + (H * V / tp) * (32 if pp == 1 else 16) * (alpha / D + 1 - alpha)
    def predict_memory(S):
        activation_memory = L * (S * H / tp) * A
        print(f"Predict [tp = {tp}, pp = {pp}, dp = {D}, seqlen = {S}] memory:")
        print(f"Activation memory: {activation_memory / (1024 * 1024)} MiB")
        print(f"Parameter & Gradient & Optimizer States memory: {p_g_os_memory / (1024 * 1024)} MiB")
        print(f"Gap (cuda & nccl context): {gap / (1024 * 1024)} MiB")
        return activation_memory + p_g_os_memory + gap
    def predict_max_seqlen():
        max_activation_memory = gpu_memory_bound - safe_bound - p_g_os_memory - gap
        max_seqlen = max_activation_memory / (L * (H / tp) * A)
        return int(max_seqlen) 
    # test
    if unit_test:
        print(predict_memory(256) / (1024 * 1024))    
    return predict_max_seqlen()

if __name__ == '__main__':
    file_path = 'strategy_pool.json'
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(strategy_max_seqlen(data, 1, 4))