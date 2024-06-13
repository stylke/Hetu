rank_to_device_mapping = {0: 10, 1: 8, 2: 2, 3: 0, 4: 16, 5: 17, 6: 24, 7: 25, 8: 12, 9: 9, 10: 4, 11: 1, 12: 18, 13: 19, 14: 26, 15: 27, 16: 11, 17: 13, 18: 3, 19: 5, 20: 20, 21: 21, 22: 28, 23: 29, 24: 14, 25: 15, 26: 6, 27: 7, 28: 22, 29: 23, 30: 30, 31: 31}
l_values = [[9, 8, 20, 23], [15, 15, 15, 15], [15, 15, 15, 15], [15, 15, 15, 15]]

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