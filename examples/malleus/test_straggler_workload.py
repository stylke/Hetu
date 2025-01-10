import hetu as ht
from hetu.engine.straggler import *

device = ht.device("cuda:1")
workload_info = WorkloadInfo(
    2,
    1024,
    4096
)
straggler = Straggler(
    device,
    "./tmp/log.txt",
    workload_info
)
straggler.begin_profile()
print(f"{device}: pre-profiling straggler workload...")
for _ in range(10):
    straggler.run_profile()
straggler.end_profile()