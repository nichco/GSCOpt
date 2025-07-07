import numpy as np
import matplotlib.pyplot as plt



copies = np.array([2, 3, 4, 5, 6, 7, 8, 9])

# monolithic_time = np.array([11.626, 42.037, 133.012, 621.379, 1467.451, 2944.026, 6370])
monolithic_time = np.array([9.86, 24.22, 112.345, 386.23, 755.009, 1504.773, 2785.930, 3508.666])
# 2: 9.86
# 3: 24.22
# 4: 112.345
# 5: 386.23
# 6: 755.009
# 7: 1504.773
# 8: 2785.930
# 9: 3508.666
# 10: 

# distributed_time = np.array([52.335, 88.211, 205.231, 161.269, 395.084, 429.445, 578.767])
distributed_time = np.array([40.675, 58.815, 176.175, 265.7, 320.679, 399.415, 565.788, 588.668])
# 2: 40.675
# 3: 58.815
# 4: 176.175
# 5: 265.7
# 6: 320.679
# 7: 399.415
# 8: 565.788
# 9: 588.668
# 10: 

plt.figure(figsize=(5, 4))
plt.rcParams.update({'font.size': 12})

plt.plot(copies, monolithic_time, label='Monolithic', marker='o', markersize=8)
plt.plot(copies, distributed_time, label='Distributed', marker='s', markersize=8)
plt.xlabel('Sub-problems')
plt.ylabel('CPU Time (s)')
plt.legend()

plt.yscale('log')
plt.savefig('timing.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()