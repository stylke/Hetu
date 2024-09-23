import re
import matplotlib.pyplot as plt

GPUS_NUM = 16

def plot_total_run_times(file_path):
    plt.figure(figsize=(12, 8))

    total_run_times = []
    estimated_times_1 = []
    estimated_times_2 = []

    # 使用正则表达式匹配 "total run time: <number> ms"
    pattern_1 = re.compile(r'total run time:\s*(\d+)\s*ms')
    pattern_2 = re.compile(r'estimated cost is\s*([\d.]+)')

    for i in range(GPUS_NUM):
        cnt = 0
        cur_total_run_times = []
        cur_estimated_times_1 = []
        cur_estimated_times_2 = []
        with open(file_path + f"log_{i}.txt", 'r') as file:
            try:
                for line in file:
                    match = pattern_1.search(line)
                    if match:
                        cur_total_run_times.append(int(match.group(1)))
                    match = pattern_2.search(line)
                    if match:
                        if cnt % 2 == 0:
                            cur_estimated_times_1.append(float(match.group(1)))
                        else:
                            cur_estimated_times_2.append(float(match.group(1)))
                        cnt += 1
            except Exception as e:
                raise e
                pass 
        # print(cur_total_run_times, cur_estimated_times_1, cur_estimated_times_2)
        total_run_times.append(cur_total_run_times[1:])
        estimated_times_1.append(cur_estimated_times_1)
        estimated_times_2.append(cur_estimated_times_2)

    max_total_run_times = []
    max_estimated_times_1 = []
    max_estimated_times_2 = []
    for j in range(80):
        max_total_run_times.append(max([total_run_times[i][j] for i in range(GPUS_NUM)]))
        max_estimated_times_1.append(max([estimated_times_1[i][j] for i in range(GPUS_NUM)]))
        max_estimated_times_2.append(max([estimated_times_2[i][j] for i in range(GPUS_NUM)]))
        
    plt.plot(max_total_run_times, marker='o', linestyle='-', label='E2E')
    plt.plot(max_estimated_times_1, marker='o', linestyle='-', label='Estimation 1')
    plt.plot(max_estimated_times_2, marker='o', linestyle='-', label='Estimation 2')

    plt.title('Estimated Time Comparison (homo strategy, gbs=64)')
    plt.xlabel('Index')
    plt.ylabel('Time Cost (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./homo_estimation.png')

if __name__ == "__main__":
    file_path = '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case3/llama7b_gpus16_gbs64_msl8192/'
    plot_total_run_times(file_path)