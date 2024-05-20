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
memory_d = [5939, 4558, 4474, 5527]
memory_bound = 40536.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 3

ctxs = TrainerCtxs(
    bf16=True,
    hetero_tp_alpha=hetero_tp_alpha,
    hetero_tp_weight=hetero_tp_weight,
    normal_layers=layers // pp,
    normal_mbn=mbn // dp,
    normal_compute_time=normal_compute_time,
    memory_k=memory_k,
    memory_d=memory_d,
    memory_bound=memory_bound,
    memory_safe_gap=memory_safe_gap,
    straggler_threshold=straggler_threshold,
    straggler_safe_gap=straggler_safe_gap,
    top_k=top_k
)

zero = True
rank_to_device_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15}
suspended_rank_list = []
unused_rank_list = []
hetero_data = True
hetero_layers =[[8,8,8,8],[8,8,8,8]]
hetero_micro_batch_num_list = [32,32]
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
    hetero_micro_batch_num_list=hetero_micro_batch_num_list
)

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
suspended_devices_sr = {
    
}
unused_devices = []
strategy_model = StrategyModel(
    ctxs,
    strategy_args,
    used_devices_sr,
    suspended_devices_sr,
    unused_devices
)
strategies, _ = strategy_model.make_plans()
Args.print_args_list(strategies)