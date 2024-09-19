import json

json_path = "strategy_pool.json"

def generate_json(data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data has been written to \"{json_path}\"")

if __name__ == '__main__':
    # 自定义数据
    hidden_size = 4096
    num_layers = 32
    vocab_size = 30592
    # tp_pp_list = [(1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)]
    # a_list = [-1.52587891e-05, -2.31879928e-06, -9.26268522e-06, -4.85685844e-07, -3.56593792e-08, -2.68655783e-06, -4.95452836e-08, 6.98499776e-08, -4.03883018e-07, 1.37774147e-07]
    # b_list = [8.01809211e-02, 3.44067829e-02, 8.59247240e-02, 3.36870955e-02, 1.66097866e-02, 8.38166218e-02, 3.63550659e-02, 1.80880312e-02, 4.83143430e-02, 2.10431364e-02]
    # c_list = [None] * 10
    tp_pp_list = [(1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (8, 1)]
    a_list = [0, 2.34319074e-06, 4.95466356e-07, 4.59723501e-07, 2.04864710e-06, 8.87526098e-07, 4.17956749e-07, 9.50733003e-07, 4.60260699e-07, 5.11853937e-07]
    b_list = [0.19650879, 8.85071471e-02, 4.94624083e-02, 2.46004879e-02, 9.72555576e-02, 5.07571080e-02, 2.63047347e-02, 5.55708100e-02, 2.82728634e-02, 3.26816250e-02]
    c_list = [33.006249999999966, 29.37029734970929, 10.561974789915922, 6.958292347039134, 52.17134541703467, 24.612925785287587, 12.269063978583347, 51.3784736430714, 26.16629859609054, 59.6554489383044]
    utilization_seqlen = {'tp1': 1280, 'tp2': 2304, 'tp4': 3072, 'tp8': 4352}
    memory_A = {'tp1': 38, 'tp2': 39, 'tp4': 42, 'tp8': 48}
    memory_B = 197
    memory_alpha = 0.5
    memory_gap = {'gpu1': 1755 * (1024 ** 2), 'gpu2': 4831 * (1024 ** 2), 'gpu4': 7299 * (1024 ** 2), 'node': 12281 * (1024 ** 2)}
    memory_safe_bound = 2 * (1024 ** 3)
    gpus_per_node = 8
    gpu_memory_bound = (40536 + 9000) * (1024 ** 2)
    # 生成json
    strategy_list = []
    data = {}
    for (tp, pp), a, b, c in zip(tp_pp_list, a_list, b_list, c_list):
        strategy_list.append({
            "tp": tp,
            "pp": pp,
            "a": a,
            "b": b,
            "c": c
        })
    data['strategies'] = strategy_list
    data['memory_regression'] = {}
    data['memory_regression']['A'] = memory_A
    data['memory_regression']['B'] = memory_B
    data['memory_regression']['alpha'] = memory_alpha
    data['memory_regression']['gap'] = memory_gap
    data['memory_regression']['safe_bound'] = memory_safe_bound
    data['model_config'] = {}
    data['model_config']['H'] = hidden_size
    data['model_config']['L'] = num_layers
    data['model_config']['V'] = vocab_size
    data['cluster_config'] = {}
    data['cluster_config']['gpus_per_node'] = gpus_per_node
    data['cluster_config']['gpu_memory_bound'] = gpu_memory_bound
    data['cluster_config']['utilization_seqlen'] = utilization_seqlen
    generate_json(data)