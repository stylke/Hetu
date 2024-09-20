import re
import matplotlib.pyplot as plt

def extract_total_run_times(file_path):
    total_run_times = []

    # 使用正则表达式匹配 "total run time: <number> ms"
    pattern = re.compile(r'total run time:\s*(\d+)\s*ms')

    with open(file_path, 'r') as file:
        try:
            for line in file:
                match = pattern.search(line)
                if match:
                    total_run_times.append(int(match.group(1)))
        except Exception:
            pass 

    return total_run_times[1:80]

def plot_total_run_times(file_paths, labels):
    plt.figure(figsize=(12, 8))

    for file_path, label in zip(file_paths, labels):
        total_run_times = extract_total_run_times(file_path)
        plt.plot(total_run_times, marker='o', linestyle='-', label=label)

    plt.title('Total Run Time Comparison (gbs=64)')
    plt.xlabel('Index')
    plt.ylabel('Total Run Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./e2e.png')

if __name__ == "__main__":
    file_paths = [
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case1/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case2/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case3/llama7b_gpus16_gbs64_msl8192/log_0.txt',
        # '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case2/llama7b_gpus16_gbs128_msl8192/log_0.txt',
        # '/home/pkuhetu/lhy/multi_switch/examples/hydraulis/logs/case3/llama7b_gpus16_gbs128_msl8192/log_0.txt'
    ]
    labels = [
        "Greedy Packing (Static Shape)",
        "Greedy Packing (Dynamic Shape)",
        "Our Packing",
        # "Balanced Packing",
        # "Ours Estimated Packing"
        # "Hetero Packing"
    ]
    plot_total_run_times(file_paths, labels)