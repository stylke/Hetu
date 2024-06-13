dp = 4
pp = 4
L = 60
M = 512

pipelines_template = [[2.349927546535091, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
objective = 3312.0
l_values = [[17, 20, 23], [17, 20, 23], [17, 20, 23], [17, 20, 23]]
m_values = [82, 143, 143, 144]

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