from elastic.engine.strategy import *
from elastic.engine.utils import *
from io import StringIO
import sys

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

# tencent 32 A800s llama 4k gbs=64 mbs=1
dp = 4
tp = 2
pp = 4
layers = 60
mbn = 512
hetero_tp_alpha = [1.0, 2.0, 4.0, 8.0]
hetero_tp_weight = [1.0, 1.2, 1.4, 1.6]
# normal_compute_time = 66111.4
normal_compute_time = 8305
memory_k = [4608.02, 4124.24, 3641.99, 3129.95]
# memory_k = [3745.47, 3389.63, 3038.79,2693.70]
memory_embedding = 1200.0
memory_extra = 5500.0
# memory_embedding = 1000.0
# memory_extra = 4600.0
# memory_d = [6837.36, 5693.32, 5440.25, 6546.01]
memory_bound = 81252.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 3
zero = True
rank_to_device_mapping = {0:15,1:8,2:7,3:0,4:16,5:17,6:24,7:25,8:10,9:9,10:2,11:1,12:18,13:19,14:26,15:27,16:12,17:11,18:4,19:3,20:20,21:21,22:28,23:29,24:14,25:13,26:6,27:5,28:22,29:23,30:30,31:31}
suspended_rank_list = [3]
unused_rank_list = []
hetero_data = True
hetero_layers = [[11,8,19,22],[15,15,15,15],[15,15,15,15],[15,15,15,15]]
hetero_stages = [4,4,4,4]
hetero_micro_batch_num_list = [10,18,18,18]

'''
# tencent 64 A800s llama 4k gbs=64 mbs=1
dp = 4
tp = 4
pp = 4
layers = 80
mbn = 64 # 512
hetero_tp_alpha = [1.0, 2.0, 4.0, 8.0]
hetero_tp_weight = [1.0, 1.2, 1.4, 1.6]
normal_compute_time = 1 # 不重要
memory_k = [4450.8, 4095.4, 3745.47, 3389.63, 3038.79, 2693.70]
memory_embedding = 1000.0
memory_extra = 4600.0
memory_bound = 81252.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 3
zero = True
rank_to_device_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,18:18,19:19,20:20,21:21,22:22,23:23,24:24,25:25,26:26,27:27,28:28,29:29,30:30,31:31,32:32,33:33,34:34,35:35,36:36,37:37,38:38,39:39,40:40,41:41,42:42,43:43,44:44,45:45,46:46,47:47,48:48,49:49,50:50,51:51,52:52,53:53,54:54,55:55,56:56,57:57,58:58,59:59,60:60,61:61,62:62,63:63}
suspended_rank_list = [1]
unused_rank_list = []
hetero_data = True
hetero_layers = [[20,20,20,20],[20,20,20,20],[20,20,20,20],[20,20,20,20]]
hetero_stages = [4,4,4,4]
hetero_micro_batch_num_list = [16,16,16,16]
'''

# tencent 64 A800s llama 70b 10k gbs=64 mbs=1
dp = 2
tp = 8
pp = 4
layers = 80
mbn = 64 # 512
hetero_tp_alpha = [1.0, 1.90, 3.66, 7.22]
hetero_tp_weight = [1.0, 1.0, 1.0, 1.0]
normal_compute_time = 1 # 不重要
memory_k = [2685, 2485, 2295, 2110, 1933, 1769, 1647]  
memory_embedding = 600.0
memory_extra = 4700.0
memory_bound = 81252.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 10
zero = True
rank_to_device_mapping = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:48,17:49,18:50,19:51,20:52,21:53,22:54,23:55,24:56,25:57,26:58,27:59,28:60,29:61,30:62,31:63,32:40,33:41,34:42,35:43,36:44,37:45,38:46,39:47,40:16,41:17,42:18,43:19,44:20,45:21,46:22,47:23,48:24,49:25,50:26,51:27,52:28,53:29,54:30,55:31,56:32,57:33,58:34,59:35,60:36,61:37,62:38,63:39}
suspended_rank_list = []
unused_rank_list = []
hetero_data = True
hetero_layers = [[3,25,26,26],[20,20,20,20]]
hetero_stages = [7,5]
hetero_micro_batch_num_list = [31,33]

'''
# tencent 64 A800s llama 110b 10k gbs=64 mbs=1
dp = 2
tp = 8
pp = 4
layers = 80
mbn = 64 # 512
hetero_tp_alpha = [1.0, 1.90, 3.66, 7.22]
hetero_tp_weight = [1.0, 1.0, 1.0, 1.0]
normal_compute_time = 1 # 不重要
memory_k = [4240, 3930, 3630, 3338, 3068, 2830, 2597, 2438]  
memory_embedding = 500.0
memory_extra = 4900.0
memory_bound = 81252.0
memory_safe_gap = 4096.0
straggler_threshold = 1.2
straggler_safe_gap = 0.3
top_k = 10
zero = True
rank_to_device_mapping = {0:7,1:0,2:8,3:64,4:65,5:66,6:67,7:68,8:5,9:6,10:69,11:70,12:71,13:72,14:73,15:74,16:1,17:2,18:3,19:4,20:75,21:76,22:77,23:78,24:40,25:41,26:42,27:43,28:44,29:45,30:46,31:47,32:16,33:17,34:18,35:19,36:20,37:21,38:22,39:23,40:24,41:25,42:26,43:27,44:28,45:29,46:30,47:31,48:15,49:79,50:80,51:81,52:82,53:83,54:84,55:85,56:13,57:14,58:86,59:87,60:88,61:89,62:90,63:91,64:9,65:10,66:11,67:12,68:92,69:93,70:94,71:95,72:32,73:33,74:34,75:35,76:36,77:37,78:38,79:39,80:48,81:49,82:50,83:51,84:52,85:53,86:54,87:55,88:56,89:57,90:58,91:59,92:60,93:61,94:62,95:63}
suspended_rank_list = [1,2]
unused_rank_list = [3,4,5,6,7,10,11,12,13,14,15,20,21,22,23,49,50,51,52,53,54,55,58,59,60,61,62,63,68,69,70,71]
hetero_data = True
hetero_layers = [[2,5,11,20,21,21],[2,5,11,20,21,21]]
hetero_stages = [6,6]
hetero_micro_batch_num_list = [32,32]
'''

# setting 1
used_devices_sr = {
    0:155357 / 66111.4,
    # 0:316734 / 66111.4,
    1:155357 / 66111.4,
    2:155357 / 66111.4,
    3:155357 / 66111.4,
    4:155357 / 66111.4,
    5:155357 / 66111.4,
    6:155357 / 66111.4,
    7:155357 / 66111.4,
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

# setting 2
used_devices_sr = {
    0:6.87,
    # 0:54372 / 9663,
    1:6.87,
    2:6.87,
    3:6.87,
    4:6.87,
    5:6.87,
    6:6.87,
    7:6.87,
    8:5.42,
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
    31:1,
    32:1,
    33:1,
    34:1,
    35:1,
    36:1,
    37:1,
    38:1,
    39:1,
    40:1,
    41:1,
    42:1,
    43:1,
    44:1,
    45:1,
    46:1,
    47:1,
    48:1,
    49:1,
    50:1,
    51:1,
    52:1,
    53:1,
    54:1,
    55:1,
    56:1,
    57:1,
    58:1,
    59:1,
    60:1,
    61:1,
    62:1,
    63:1,
}
suspended_devices_sr = {}
unused_devices = []

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
# 创建一个StringIO对象来捕获输出
output_capture = StringIO()
# 保存原来的标准输出
original_stdout = sys.stdout
try:
    # 将标准输出重定向到StringIO对象
    sys.stdout = output_capture
    # 这里的print会被捕获到StringIO对象中
    Args.print_args_list(strategies)
finally:
    # 恢复标准输出
    sys.stdout = original_stdout

# 获取捕获的输出并将其转换为字符串
output_str = output_capture.getvalue()
print(output_str.replace(" ", ""))
