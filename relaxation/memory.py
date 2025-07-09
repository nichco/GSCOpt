import numpy as np
import matplotlib.pyplot as plt



copies = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])

# monolithic_memory_csdl = np.array([53.302, 73.99, 100.646, 132.946, 170.566, 213.768, 262.973, 318.20, 377.307]) - 47.98 # (the 40 is the csdl cost...)
monolithic_memory_jax = np.array([11.063, 21.581, 35.208, 52.54, 73.571, 98.296, 126.656, 158.705, 193.573])

distributed_memory = np.array([5.006, 5.116, 5.33, 5.497, 5.641, 5.864, 6.134, 6.368, 6.657])




plt.figure(figsize=(5, 4))
plt.rcParams.update({'font.size': 12})

plt.plot(copies, monolithic_memory_jax, label='Monolithic', marker='o', markersize=8)
plt.plot(copies, distributed_memory, label='Distributed', marker='s', markersize=8)
plt.xlabel('Number of sub-problems')
plt.ylabel('Memory (MB)')
plt.legend()
plt.yscale('log')

plt.grid(True, which="both", linestyle='-', linewidth=0.5, alpha=0.3)

plt.savefig('memory.png', dpi=300, transparent=True, bbox_inches='tight')
plt.show()