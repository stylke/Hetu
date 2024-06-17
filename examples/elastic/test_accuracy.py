dp = 2
pp = 4
L = 80
M = 64

pipelines_template = [[7.22, 7.22, 7.22, 1.9, 1.9, 1.9, 1.0, 1.0], [3.66, 3.66, 3.66, 1.0, 1.0, 1.0]]
pipelines_template_devices = [[[7], [15], [23], [1, 2, 3, 4], [9, 10, 11, 12], [17, 18, 19, 20], None, None], [[5, 6], [13, 14], [21, 22], None, None, None]]
objective = 693.0
l_values = [[2, 2, 2, 10, 11, 11, 21, 21], [4, 5, 5, 22, 22, 22]]
m_values = [33, 31]

'''
dp = 4
pp = 4
L = 60
M = 64

pipelines_template = [[2.34, 3.8, 1.0, 1.0], [2.34, 1.0, 1.0, 1.0], [2.34, 1.0, 1.0, 1.0], [2.34, 1.0, 1.0, 1.0]]
l_values = [[12.0, 7.0, 19.0, 22.0], [7.0, 17.0, 18.0, 18.0], [7.0, 17.0, 18.0, 18.0], [7.0, 17.0, 18.0, 18.0]]
m_values = [11.0, 18.0, 17.0, 18.0]
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