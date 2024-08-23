import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def process_and_plot(file_paths, names, ranges):
    colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive']  # 预定义颜色列表
    plt.figure()

    for idx, file_path in enumerate(file_paths):
        # 读取txt文件内容
        with open(file_path, 'r') as file:
            data = file.read()

        # 使用正则表达式提取seq_len和时间数据
        seq_lens = [1024 * int(x) for x in re.findall(r'packing num = (\d+):', data)]
        times = [float(x) for x in re.findall(r'(\d+\.\d+)s', data)]

        seq_lens = seq_lens[:int(ranges[idx] * len(seq_lens))]
        times = times[:int(ranges[idx] * len(times))]
        
        # 转换为numpy数组
        X = np.array(seq_lens).reshape(-1, 1)
        y = np.array(times)

        # 使用线性回归进行拟合
        model = LinearRegression(fit_intercept=False)  # 不包括截距
        model.fit(X, y)

        # 获取一次函数的系数
        coefficient = model.coef_[0]

        # 打印一次函数的系数
        print(f'文件 {file_path} 的一次函数系数: {coefficient}')

        # 生成回归曲线数据
        X_fit = np.linspace(min(seq_lens), max(seq_lens), 100).reshape(-1, 1)
        y_fit = model.predict(X_fit)

        # 绘制回归曲线和数据点
        plt.scatter(seq_lens, times, color=colors[idx % len(colors)], alpha=0.6)  # 点没有label
        plt.plot(X_fit, y_fit, color=colors[idx % len(colors)], label=names[idx])

    plt.xlabel('sequence length (token)')
    plt.ylabel('time (s)')
    plt.ylim(0, 45)
    plt.title('Pure Linear Regression of Variable Length Attention')
    plt.legend()
    plt.savefig("./fig_packing.png")

# 示例调用
file_paths = ['./packing_1024/num_heads_4.txt', './packing_1024/num_heads_8.txt', './packing_1024/num_heads_16.txt', './packing_1024/num_heads_32.txt']  # 添加更多文件路径
# ranges = [1, 1, 0.75, 0.5]
ranges = [1, 1, 1, 1]
names = ["TP=8", "TP=4", "TP=2", "TP=1"]
process_and_plot(file_paths, names, ranges)
