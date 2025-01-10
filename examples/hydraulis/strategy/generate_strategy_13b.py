import json

json_path = "strategy_pool_13b.json"

def generate_json(data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data has been written to \"{json_path}\"")

if __name__ == '__main__':
    # 自定义数据
    hidden_size = 5120
    num_layers = 40
    vocab_size = 30592
    # tp_pp_list = [(1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)]
    # a_list = [-1.52587891e-05, -2.31879928e-06, -9.26268522e-06, -4.85685844e-07, -3.56593792e-08, -2.68655783e-06, -4.95452836e-08, 6.98499776e-08, -4.03883018e-07, 1.37774147e-07]
    # b_list = [8.01809211e-02, 3.44067829e-02, 8.59247240e-02, 3.36870955e-02, 1.66097866e-02, 8.38166218e-02, 3.63550659e-02, 1.80880312e-02, 4.83143430e-02, 2.10431364e-02]
    # c_list = [None] * 10
    tp_pp_list = [
        (1, 1), (1, 2), (1, 4), (1, 8),
        (2, 1), (2, 2), (2, 4), (2, 8),
        (4, 1), (4, 2), (4, 4),
        (8, 1)
    ]
    a_list = [
        1.03359006e-05, 2.33212956e-06, 1.25504641e-06, 7.57890093e-07,
        2.93719955e-06, 1.53436151e-06, 7.97420826e-07, 4.03752453e-07,
        1.62871921e-06, 8.12710963e-07, 3.94357054e-07,
        6.59233262e-07
    ]
    b_list = [
        3.12262914e-01, 1.68111063e-01, 8.51150701e-02, 4.24526643e-02,
        1.80090146e-01, 9.00918438e-02, 4.52101722e-02, 2.33333496e-02,
        9.97626395e-02, 5.00755835e-02, 2.59475095e-02,
        6.66817542e-02
    ]
    c_list = [
        91.37538699690367, 37.86567681070392, 20.19215142652746, 13.324264844697922,
        75.87395881841167, 40.48058869844306, 23.26089691411994, 11.803803239246747,
        76.43818547098908, 39.53936966798551, 18.2104987061378,
        50.37694379271386
    ]

    utilization_seqlen = {'tp1': 640, 'tp2': 896, 'tp4': 1280, 'tp8': 1536}
    memory_A = {'tp1': 38, 'tp2': 39, 'tp4': 39, 'tp8': 48}
    memory_B = 210
    memory_alpha = 0.75
    memory_gap = {'gpu1': 5000 * (1024 ** 2), 'gpu2': 5000 * (1024 ** 2), 'gpu4': 5000 * (1024 ** 2), 'node': 5000 * (1024 ** 2)}
    memory_safe_bound = 8 * (1024 ** 3)
    gpus_per_node = 8
    gpu_memory_bound = (81252) * (1024 ** 2)
    hetero_dp_comm_cost = 300 # ms
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
    data['comm_cost'] = {}
    data['comm_cost']['hetero_dp'] = hetero_dp_comm_cost
    generate_json(data)