from elastic.engine.strategy import *
from elastic.engine.utils import *

dp = 2
tp = 2
pp = 4
layers = 32
mbn = 64
hetero_tp_alpha = [1.0, 2.0, 4.0, 8.0]
hetero_tp_weight = [1.0, 1.0, 1.0, 1.0]
normal_compute_time = 4000.0
memory_k = [2934, 2483, 2024, 1567]
memory_embedding = 1200.0
memory_extra = 4500.0
# memory_d = [5939, 4558, 4474, 5527]
memory_bound = 40536.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 3
zero = True
rank_to_device_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}
suspended_rank_list = []
unused_rank_list = []
hetero_data = True
hetero_layers = [[8,8,8,8],[8,8,8,8]]
hetero_stages = [4,4]
hetero_micro_batch_num_list = [32,32]

# setting 1
used_devices_sr = {
    0:1.5,
    1:1,
    2:1,
    3:1,
    4:1,
    5:1.1,
    6:1,
    7:1,
    8:1,
    9:1,
    10:1,
    11:0.8,
    12:1.2,
    13:1,
    14:1,
    15:5.0
}
suspended_devices_sr = {}
unused_devices = []

# setting 2
used_devices_sr = {
    8:1,
    9:1,
    10:1,
    11:0.8,
    12:1.2,
    13:1,
    14:1,
    15:5.0
}
suspended_devices_sr = {}
unused_devices = [0, 1, 2, 3, 4, 5, 6, 7]

# setting 3
used_devices_sr = {
    0:1.5,
    2:1,
    3:1,
    4:2.0,
    5:1.1,
    6:1,
    7:3.0,
    8:1,
    9:1,
    10:1,
    11:0.8,
    12:1.2,
    13:1,
    14:1,
    15:5.0
}
suspended_devices_sr = {
    1:1.1,
}
unused_devices = []

# tencent 32 A800s llama 4k gbs=512 mbs=1
dp = 4
tp = 2
pp = 4
layers = 60
mbn = 64 # 512
hetero_tp_alpha = [1.0, 2.0, 4.0, 8.0]
hetero_tp_weight = [1.0, 1.2, 1.4, 1.6]
normal_compute_time = 66111.4
memory_k = [4608.02, 4124.24, 3641.99, 3129.95]
# memory_k = [3745.47, 3389.63, 3038.79,2693.70]
memory_embedding = 1200.0
memory_extra = 5500.0
# memory_embedding = 1000.0
# memory_extra = 4600.0
# memory_d = [6837.36, 5693.32, 5440.25, 6546.01]
memory_bound = 81252.0
memory_safe_gap = 2048.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 3
zero = True
rank_to_device_mapping = {0:0,1:1,2:8,3:9,4:16,5:17,6:24,7:25,8:2,9:3,10:10,11:11,12:18,13:19,14:26,15:27,16:4,17:5,18:12,19:13,20:20,21:21,22:28,23:29,24:6,25:7,26:14,27:15,28:22,29:23,30:30,31:31}
suspended_rank_list = []
unused_rank_list = []
hetero_data = True
hetero_layers = [[15,15,15,15],[15,15,15,15],[15,15,15,15],[15,15,15,15]]
hetero_stages = [4,4,4,4]
hetero_micro_batch_num_list = [128,128,128,128]

# setting 1
used_devices_sr = {
    # 0:155357 / 66111.4,
    0:316734 / 66111.4,
    1:1,
    2:1,
    3:1,
    4:1,
    5:1,
    6:1,
    7:1,
    8:1,
    9:1,
    10:1,
    11:1,
    12:1,
    13:1,
    14:1,
    15:1,
    16:1,
    17:1,
    18:1,
    19:1,
    20:1,
    21:1,
    22:1,
    23:1,
    24:1,
    25:1,
    26:1,
    27:1,
    28:1,
    29:1,
    30:1,
    31:1
}
suspended_devices_sr = {}
unused_devices = []

'''
# setting 2
used_devices_sr = {
    8:2.34,
    9:1,
    10:1,
    11:1,
    12:1,
    13:1,
    14:1,
    15:1,
    16:1,
    17:1,
    18:1,
    19:1,
    20:1,
    21:1,
    22:1,
    23:1,
    24:1,
    25:1,
    26:1,
    27:1,
    28:1,
    29:1,
    30:1,
    31:1
}
suspended_devices_sr = {}
unused_devices = [0,1,2,3,4,5,6,7]
'''

ctxs = TrainerCtxs(
    bf16=True,
    hetero_tp_alpha=hetero_tp_alpha,
    hetero_tp_weight=hetero_tp_weight,
    normal_layers=layers // pp,
    normal_mbn=mbn // dp,
    normal_compute_time=normal_compute_time,
    memory_k=memory_k,
    memory_extra=memory_extra,
    memory_embedding=memory_embedding,
    # memory_d=memory_d,
    memory_bound=memory_bound,
    memory_safe_gap=memory_safe_gap,
    straggler_threshold=straggler_threshold,
    straggler_safe_gap=straggler_safe_gap,
    top_k=top_k
)

strategy_args = TrainerStrategyArgs(
    dp=dp,
    tp=tp,
    pp=pp,
    zero=zero,
    rank_to_device_mapping=rank_to_device_mapping,
    suspended_rank_list=suspended_rank_list,
    unused_rank_list=unused_rank_list,
    hetero_data=hetero_data,
    hetero_layers=hetero_layers,
    hetero_stages=hetero_stages,
    hetero_micro_batch_num_list=hetero_micro_batch_num_list
)

strategy_model = StrategyModel(
    ctxs,
    strategy_args,
    used_devices_sr,
    suspended_devices_sr,
    unused_devices
)

strategies, _ = strategy_model.make_plans()
print("**********************************************") 
Args.print_args_list(strategies)