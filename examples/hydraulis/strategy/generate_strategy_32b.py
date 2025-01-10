import json

json_path = "strategy_pool_32b.json"

def generate_json(data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"JSON data has been written to \"{json_path}\"")

if __name__ == '__main__':
    # 自定义数据
    hidden_size = 6656
    num_layers = 60
    vocab_size = 30592
    # tp_pp_list = [(1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)]
    # a_list = [-1.52587891e-05, -2.31879928e-06, -9.26268522e-06, -4.85685844e-07, -3.56593792e-08, -2.68655783e-06, -4.95452836e-08, 6.98499776e-08, -4.03883018e-07, 1.37774147e-07]
    # b_list = [8.01809211e-02, 3.44067829e-02, 8.59247240e-02, 3.36870955e-02, 1.66097866e-02, 8.38166218e-02, 3.63550659e-02, 1.80880312e-02, 4.83143430e-02, 2.10431364e-02]
    # c_list = [None] * 10
    '''
    tp_pp_list = [(1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2)]
    a_list = [0, 9.39192752e-07, 2.97455561e-05, 3.73416233e-06, 1.97033746e-06, 3.86950049e-06, 2.04563898e-06, 1.05779156e-06, 2.12548718e-06, 1.06789378e-06]
    b_list = [0.48988971, 2.12596087e-01, 3.86957078e-01, 2.13060059e-01, 1.06558744e-01, 2.38996833e-01, 1.18301177e-01, 5.92119769e-02, 1.41411593e-01, 7.10582460e-02]
    c_list = [22.882352941176805, 35.99468155683377, 156.2008928571429, 80.71665943745563, 42.94488582340546, 120.69476474631483, 65.4239195557368, 35.33534922208071, 122.91021345581066, 61.76782590869129]
    '''
    tp_pp_list = [
        (1, 2), (1, 4),
        (2, 1), (2, 2), (2, 4), (2, 6),
        (4, 1), (4, 2), (4, 4), (4, 6),
        (8, 1), (8, 2), (8, 3), (8, 4),
        (16, 1)
    ]
    a_list = [
        0, 9.39192752e-07,
        2.97455561e-05, 3.73416233e-06, 1.97033746e-06, 1.31605318e-06,
        3.86950049e-06, 2.04563898e-06, 1.05779156e-06, 7.02993498e-07,
        2.12548718e-06, 1.06789378e-06, 6.85518086e-07, 4.91076016e-07,
        1.37779723e-06
    ]
    b_list = [
        0.48988971, 2.12596087e-01,
        3.86957078e-01, 2.13060059e-01, 1.06558744e-01, 7.18051676e-02,
        2.38996833e-01, 1.18301177e-01, 5.92119769e-02, 3.98508980e-02,
        1.41411593e-01, 7.10582460e-02, 4.82754236e-02, 3.70804063e-02,
        1.02435387e-01
    ]
    c_list = [
        22.882352941176805, 35.99468155683377,
        156.2008928571429, 80.71665943745563, 42.94488582340546, 28.616273292576068,
        120.69476474631483, 65.4239195557368, 35.33534922208071, 24.07199162618781,
        122.91021345581066, 61.76782590869129, 38.06972982282218, 25.442409002069667,
        145.74761115393085
    ]

    # utilization_seqlen = {'tp1': 640, 'tp2': 1024, 'tp4': 1536, 'tp8': 2432, 'tp16': 3968}
    utilization_seqlen = {'tp1': 384, 'tp2': 512, 'tp4': 768, 'tp8': 1024, 'tp16': 1280}
    memory_A = {'tp1': 36, 'tp2': 36, 'tp4': 37, 'tp8': 42, 'tp16': 50}
    memory_B = 201
    memory_alpha = 0.75
    memory_gap = {'gpu1': 5000 * (1024 ** 2), 'gpu2': 5000 * (1024 ** 2), 'gpu4': 5000 * (1024 ** 2), 'node': 5000 * (1024 ** 2)}
    memory_safe_bound = 10 * (1024 ** 3)
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