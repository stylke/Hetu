dp = 2
pp = 4
L = 80
M = 64

pipelines_template = [[7.22, 3.66, 3.66, 1.9, 1.9, 1.9, 1.0, 1.0], [7.22, 7.22, 3.66, 1.0, 1.0, 1.0]]
pipelines_template_devices = [[[7], [5, 6], [13, 14], [1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20], None, None], [[15], [23], [21, 22], None, None, None]]
objective = 667.0
l_values = [[2, 5, 5, 10, 10, 10, 19, 19], [3, 3, 6, 23, 22, 23]]
m_values = [35, 29]

'''
dp = 4
pp = 4
L = 60
M = 64

pipelines_template = [[2.34, 2.0, 1.0, 1.0], [3.8, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
pipelines_template_devices = [[[7, 0], [23], None, None], [[15, 8], None, None, None], [None, None, None, None], [None, None, None, None]]
objective = 285.0
l_values = [[11, 8, 19, 22], [5, 17, 19, 19], [15, 15, 15, 15], [15, 15, 15, 15]]
m_values = [11, 15, 19, 19]
'''

ans_1 = 0
ans_2 = 0
ans_3 = 0
all = 0
for i in range(dp):
    row = 0
    for y, l in zip(pipelines_template[i], l_values[i]):
        row += 1 / y
        all += 1 / y
        ans_1 = max(ans_1, y * l * m_values[i])
    ans_2 = max(ans_2, m_values[i] * L * 1 / row)
ans_3 = 1 / all * L * M
ans = M * L / dp / pp
        
print(ans_1 / ans, ans_2 / ans, ans_3 / ans)