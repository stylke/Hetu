rank_to_device_mapping = {0: 0, 1: 1, 2: 8, 3: 9, 4: 16, 5: 17, 6: 24, 7: 25, 8: 2, 9: 3, 10: 10, 11: 11, 12: 18, 13: 19, 14: 26, 15: 27, 16: 4, 17: 5, 18: 12, 19: 13, 20: 20, 21: 21, 22: 28, 23: 29, 24: 6, 25: 7, 26: 14, 27: 15, 28: 22, 29: 23, 30: 30, 31: 31}
l_values = [[7, 17, 18, 18], [15, 15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15]]

ans = ""
for k, v in rank_to_device_mapping.items():
    ans += str(k) + ":" + str(v) + ","
    
print("{" + ans[:-1] + "}")

ans = ""
for y in l_values:
    for x in y:
        ans += str(x) + ","
    
print(ans[:-1])

'''
ans = ""
for k in range(64):
    ans += str(k) + ":" + str(k) + ","
    
print("{" + ans[:-1] + "}")
'''