import numpy as np
import matplotlib.pyplot as plt



copies = np.array([2, 3, 4, 5, 6, 7, 8])

monolithic_time_slsqp = np.array([11.626, 42.037, 133.012, 621.379, 1467.451, 2944.026, 6000])
monolithic_time_ipopt = np.array([12.030, 26.849, 55.451, 93.275, 176.481, 400.02, ])

distributed_time_slsqp = np.array([52.335, 88.211, 205.231, 161.269, 395.084, 429.445, 578.767])

plt.figure(figsize=(5, 4))
plt.rcParams.update({'font.size': 12})

plt.plot(copies, monolithic_time_slsqp, label='Monolithic', marker='o')
plt.plot(copies, distributed_time_slsqp, label='Distributed', marker='s')
plt.xlabel('Sub-problems')
plt.ylabel('CPU Time (s)')
plt.legend()

plt.yscale('log')
plt.savefig('timing.png', dpi=300, transparent=True)
plt.show()